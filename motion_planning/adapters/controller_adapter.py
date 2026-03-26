from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from motion_planning.profiles import CONCRETE_PZS100_PROFILE, TIMBER_COMPAT_PROFILE
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


KNOWN_PROFILES = (
    CONCRETE_PZS100_PROFILE,
    TIMBER_COMPAT_PROFILE,
)


@dataclass(frozen=True)
class JointSampleMaps:
    positions: Mapping[str, float]
    velocities: Mapping[str, float]
    accelerations: Mapping[str, float]


def infer_robot_profile(joint_names: Sequence[str] | None):
    if not joint_names:
        return None
    names = tuple(str(name) for name in joint_names)
    for profile in KNOWN_PROFILES:
        if names == profile.full_state_joint_names:
            return profile
        if names == profile.command_joint_names:
            return profile
    return None


def profile_command_joint_names(joint_names: Sequence[str] | None) -> tuple[str, ...]:
    profile = infer_robot_profile(joint_names)
    if profile is not None:
        return profile.command_joint_names
    return tuple(str(name) for name in (joint_names or ()))


def project_positions_to_command_joints(
    joint_names: Sequence[str],
    positions: Mapping[str, float],
) -> dict[str, float]:
    return {
        name: float(positions[name])
        for name in profile_command_joint_names(joint_names)
        if name in positions
    }


def _resolve_value(
    target_name: str,
    source_names: Sequence[str],
    values: Mapping[str, float],
    latest_positions: Mapping[str, float],
    default: float,
    source_profile,
    target_profile,
) -> float:
    if target_name in values:
        return float(values[target_name])
    compat_source = next(
        (alias for alias, canonical in target_profile.compat_aliases.items() if canonical == target_name),
        None,
    )
    if compat_source and compat_source in values:
        return float(values[compat_source])
    if target_name in latest_positions:
        return float(latest_positions[target_name])
    if compat_source and compat_source in latest_positions:
        return float(latest_positions[compat_source])
    if source_profile is not None and target_name in source_profile.compat_aliases:
        mapped = source_profile.compat_aliases[target_name]
        if mapped in values:
            return float(values[mapped])
        if mapped in latest_positions:
            return float(latest_positions[mapped])
    if target_name in target_profile.mimic_joint_map:
        source = target_profile.mimic_joint_map[target_name]
        if source in values:
            return float(values[source])
        if source in latest_positions:
            return float(latest_positions[source])
    return float(default)


def expand_point_to_profile(
    source_joint_names: Sequence[str],
    sample_maps: JointSampleMaps,
    latest_positions: Mapping[str, float],
    target_profile=CONCRETE_PZS100_PROFILE,
) -> tuple[list[float], list[float], list[float]]:
    source_profile = infer_robot_profile(source_joint_names)
    positions_out: list[float] = []
    velocities_out: list[float] = []
    accelerations_out: list[float] = []

    current_tip = float(latest_positions.get("theta6_tip_joint", 0.0))
    current_tilt = float(latest_positions.get("theta7_tilt_joint", 0.0))

    for target_name in target_profile.full_state_joint_names:
        default_position = 0.0
        if target_name == "theta6_tip_joint":
            default_position = current_tip
        elif target_name == "theta7_tilt_joint":
            default_position = current_tilt

        positions_out.append(
            _resolve_value(
                target_name,
                source_joint_names,
                sample_maps.positions,
                latest_positions,
                default_position,
                source_profile,
                target_profile,
            )
        )
        velocities_out.append(
            _resolve_value(
                target_name,
                source_joint_names,
                sample_maps.velocities,
                {},
                0.0,
                source_profile,
                target_profile,
            )
        )
        accelerations_out.append(
            _resolve_value(
                target_name,
                source_joint_names,
                sample_maps.accelerations,
                {},
                0.0,
                source_profile,
                target_profile,
            )
        )
    return positions_out, velocities_out, accelerations_out


def densify_trajectory_for_streaming(
    trajectory: JointTrajectory,
    sample_period_s: float,
) -> JointTrajectory:
    if len(trajectory.points) <= 1:
        return trajectory

    sample_period_s = max(0.01, float(sample_period_s))
    dense = JointTrajectory()
    dense.header = trajectory.header
    dense.joint_names = list(trajectory.joint_names)

    for segment_index in range(len(trajectory.points) - 1):
        start = trajectory.points[segment_index]
        end = trajectory.points[segment_index + 1]
        start_time_s = _duration_to_seconds(start.time_from_start)
        end_time_s = _duration_to_seconds(end.time_from_start)
        duration_s = max(1e-6, end_time_s - start_time_s)
        num_steps = max(1, int(round(duration_s / sample_period_s)))

        for step in range(num_steps):
            if segment_index > 0 and step == 0:
                continue
            alpha = float(step) / float(num_steps)
            point = JointTrajectoryPoint()
            point.time_from_start = _seconds_to_duration(start_time_s + alpha * duration_s)
            point.positions = _interpolate_vector(start.positions, end.positions, alpha)
            point.velocities = _segment_velocity(start.positions, end.positions, duration_s)
            if start.accelerations and end.accelerations:
                point.accelerations = _interpolate_vector(
                    start.accelerations,
                    end.accelerations,
                    alpha,
                )
            dense.points.append(point)

    final_point = JointTrajectoryPoint()
    final_point.time_from_start = trajectory.points[-1].time_from_start
    final_point.positions = [float(v) for v in trajectory.points[-1].positions]
    if trajectory.points[-1].velocities:
        final_point.velocities = [0.0] * len(trajectory.points[-1].velocities)
    if trajectory.points[-1].accelerations:
        final_point.accelerations = [0.0] * len(trajectory.points[-1].accelerations)
    dense.points.append(final_point)
    return dense


def _interpolate_vector(start: Sequence[float], end: Sequence[float], alpha: float) -> list[float]:
    if not start or not end:
        return []
    return [
        (1.0 - alpha) * float(start[idx]) + alpha * float(end[idx])
        for idx in range(min(len(start), len(end)))
    ]


def _segment_velocity(
    start_positions: Sequence[float],
    end_positions: Sequence[float],
    duration_s: float,
) -> list[float]:
    if not start_positions or not end_positions:
        return []
    return [
        (float(end_positions[idx]) - float(start_positions[idx])) / duration_s
        for idx in range(min(len(start_positions), len(end_positions)))
    ]


def _duration_to_seconds(duration_msg) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) / 1e9


def _seconds_to_duration(seconds: float):
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    from builtin_interfaces.msg import Duration

    return Duration(sec=sec, nanosec=nanosec)
