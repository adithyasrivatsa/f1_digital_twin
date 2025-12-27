# Tests for bicycle model racing environment

import pytest
import numpy as np
from src.env.bicycle_env import BicycleRacingEnv, BicycleEnvConfig
from src.env.vehicle import BicycleVehicle, VehicleParams, VehicleState
from src.env.track import Track, create_oval_track, create_figure_eight_track


class TestTrack:
    
    def test_oval_track_creation(self):
        """Oval track should be created with correct properties."""
        track = create_oval_track(length=1000, width=12)
        
        assert track.length > 900  # Approximately correct length
        assert track.n_points > 100
        assert np.all(track.width_left == 6.0)
        assert np.all(track.width_right == 6.0)
    
    def test_track_point_interpolation(self):
        """Track point interpolation should work."""
        track = create_oval_track()
        
        tp = track.get_point_at_s(0.0)
        assert tp.s == 0.0
        
        tp_mid = track.get_point_at_s(track.length / 2)
        assert 0 < tp_mid.s < track.length
    
    def test_world_to_frenet(self):
        """World to Frenet conversion should work."""
        track = create_oval_track()
        
        # Point on centerline at start
        tp = track.get_point_at_s(0.0)
        s, n, heading = track.world_to_frenet(tp.x, tp.y)
        
        assert abs(n) < 1.0  # Should be near centerline
    
    def test_frenet_to_world_roundtrip(self):
        """Frenet to world and back should be consistent."""
        track = create_oval_track()
        
        s_orig = 100.0
        n_orig = 2.0
        
        x, y = track.frenet_to_world(s_orig, n_orig)
        s_back, n_back, _ = track.world_to_frenet(x, y)
        
        assert abs(s_back - s_orig) < 5.0  # Allow some error
        assert abs(n_back - n_orig) < 1.0
    
    def test_curvature_ahead(self):
        """Curvature lookahead should return correct shape."""
        track = create_oval_track()
        
        distances = np.array([10, 20, 50, 100])
        curvatures = track.get_curvature_ahead(0.0, distances)
        
        assert curvatures.shape == (4,)
        assert not np.any(np.isnan(curvatures))


class TestBicycleVehicle:
    
    @pytest.fixture
    def vehicle(self):
        return BicycleVehicle()
    
    def test_initial_state(self, vehicle):
        """Vehicle should have valid initial state."""
        vehicle.reset(x=0, y=0, heading=0, speed=20)
        
        assert vehicle.state.x == 0
        assert vehicle.state.y == 0
        assert vehicle.state.vx == 20
        assert vehicle.state.vy == 0
    
    def test_straight_line_motion(self, vehicle):
        """Vehicle should move forward with throttle."""
        vehicle.reset(x=0, y=0, heading=0, speed=20)
        
        for _ in range(100):
            vehicle.step(steer_input=0, throttle=0.5, brake=0, dt=0.02)
        
        # Should have moved forward
        assert vehicle.state.x > 40  # At least 2 seconds of motion
        assert abs(vehicle.state.y) < 1.0  # Minimal lateral drift
    
    def test_steering_causes_turn(self, vehicle):
        """Steering should cause vehicle to turn."""
        vehicle.reset(x=0, y=0, heading=0, speed=30)
        
        for _ in range(100):
            vehicle.step(steer_input=0.3, throttle=0.3, brake=0, dt=0.02)
        
        # Should have turned (heading changed)
        assert abs(vehicle.state.heading) > 0.1
        # Should have moved laterally
        assert abs(vehicle.state.y) > 1.0
    
    def test_braking_reduces_speed(self, vehicle):
        """Braking should reduce speed."""
        vehicle.reset(x=0, y=0, heading=0, speed=50)
        initial_speed = vehicle.state.speed
        
        for _ in range(50):
            vehicle.step(steer_input=0, throttle=0, brake=1.0, dt=0.02)
        
        assert vehicle.state.speed < initial_speed * 0.5
    
    def test_speed_limit(self, vehicle):
        """Speed should not exceed maximum."""
        vehicle.reset(x=0, y=0, heading=0, speed=90)
        
        for _ in range(200):
            vehicle.step(steer_input=0, throttle=1.0, brake=0, dt=0.02)
        
        assert vehicle.state.vx <= vehicle.params.max_speed
    
    def test_slip_angle_computation(self, vehicle):
        """Slip angle should be computed correctly."""
        vehicle.reset(x=0, y=0, heading=0, speed=30)
        
        # Induce some lateral velocity
        vehicle.state.vy = 2.0
        
        slip = vehicle.state.slip_angle
        assert abs(slip) > 0.01  # Should have non-zero slip


class TestBicycleRacingEnv:
    
    @pytest.fixture
    def env(self):
        config = BicycleEnvConfig(
            track_name="oval",
            max_episode_steps=500,
        )
        return BicycleRacingEnv(config=config)
    
    def test_observation_space(self, env):
        """Observation space should be correct."""
        assert env.observation_space == (34,)
    
    def test_action_space(self, env):
        """Action space should be correct."""
        assert env.action_space == (3,)
    
    def test_action_bounds(self, env):
        """Action bounds should be correct."""
        low, high = env.action_bounds
        assert low.shape == (3,)
        assert high.shape == (3,)
        assert low[0] == -1.0  # steering
        assert high[0] == 1.0
        assert low[1] == 0.0   # throttle
        assert high[1] == 1.0
    
    def test_reset_returns_observation(self, env):
        """Reset should return valid observation."""
        obs, info = env.reset()
        
        assert obs.shape == (34,)
        assert not np.any(np.isnan(obs))
        assert isinstance(info, dict)
    
    def test_reset_with_seed_deterministic(self, env):
        """Reset with same seed should be deterministic."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.allclose(obs1, obs2)
    
    def test_step_returns_correct_types(self, env):
        """Step should return correct types."""
        env.reset()
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        result = env.step(action)
        
        assert result.observation.shape == (34,)
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.info, dict)
    
    def test_progress_with_throttle(self, env):
        """Vehicle should make progress with throttle."""
        env.reset()
        
        initial_progress = 0.0
        action = np.array([0.0, 0.8, 0.0], dtype=np.float32)
        
        for _ in range(100):
            result = env.step(action)
        
        assert result.info["lap_progress"] > initial_progress
    
    def test_off_track_termination(self, env):
        """Episode should terminate when off track."""
        env.reset()
        
        # Steer hard to go off track
        action = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        
        terminated = False
        for _ in range(500):
            result = env.step(action)
            if result.terminated:
                terminated = True
                break
        
        assert terminated
        assert result.info.get("off_track", False)
    
    def test_max_steps_truncation(self, env):
        """Episode should truncate at max steps."""
        config = BicycleEnvConfig(
            track_name="oval",
            max_episode_steps=50,
            off_track_terminate=False,  # Don't terminate on off-track
        )
        env = BicycleRacingEnv(config=config)
        env.reset()
        
        action = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        
        for i in range(100):
            result = env.step(action)
            if result.truncated:
                break
        
        assert result.truncated
    
    def test_observation_normalization(self, env):
        """Observations should be roughly normalized."""
        env.reset()
        
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        
        for _ in range(50):
            result = env.step(action)
        
        obs = result.observation
        
        # Most values should be in reasonable range
        assert np.all(np.abs(obs) < 10.0)
    
    def test_reward_is_finite(self, env):
        """Reward should always be finite."""
        env.reset()
        
        actions = [
            np.array([0.0, 0.5, 0.0]),
            np.array([0.5, 0.8, 0.0]),
            np.array([-0.5, 0.3, 0.5]),
        ]
        
        for _ in range(100):
            action = actions[np.random.randint(len(actions))]
            result = env.step(action.astype(np.float32))
            
            assert np.isfinite(result.reward)
            if result.terminated or result.truncated:
                break


class TestBicycleEnvIntegration:
    """Integration tests with training infrastructure."""
    
    def test_gym_wrapper_compatibility(self):
        """Bicycle env should work with GymWrapper."""
        from src.env import GymWrapper
        
        config = BicycleEnvConfig(track_name="oval", max_episode_steps=100)
        env = BicycleRacingEnv(config=config)
        wrapped = GymWrapper(env)
        
        obs, info = wrapped.reset()
        assert obs.shape == wrapped.observation_space.shape
        
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        
        assert obs.shape == wrapped.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_vectorized_env(self):
        """Bicycle env should work in vectorized setting."""
        from src.env import make_vec_env
        
        config = {
            "env": {
                "name": "bicycle",
                "max_episode_steps": 100,
                "track_name": "oval",
            },
            "state": {"dimension": 34},
            "action": {"dimension": 3},
        }
        
        envs = make_vec_env(config, num_envs=2)
        
        obs, _ = envs.reset()
        assert obs.shape == (2, 34)
        
        actions = np.zeros((2, 3), dtype=np.float32)
        actions[:, 1] = 0.5  # Some throttle
        
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        assert obs.shape == (2, 34)
        
        envs.close()
