# Tests for stub environment

import pytest
import numpy as np
from src.env import StubRacingEnv, make_env, GymWrapper


class TestStubRacingEnv:
    
    @pytest.fixture
    def env(self):
        return StubRacingEnv()
    
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
        """Reset should return observation and info."""
        obs, info = env.reset()
        assert obs.shape == (34,)
        assert isinstance(info, dict)
    
    def test_reset_with_seed(self, env):
        """Reset with seed should be deterministic."""
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
    
    def test_step_invalid_action_shape(self, env):
        """Step should reject invalid action shape."""
        env.reset()
        with pytest.raises(AssertionError):
            env.step(np.array([0.0, 0.5]))  # Wrong shape
    
    def test_episode_terminates_off_track(self, env):
        """Episode should terminate when off track."""
        env.reset()
        
        # Steer hard left repeatedly
        action = np.array([-1.0, 0.5, 0.0], dtype=np.float32)
        
        terminated = False
        for _ in range(1000):
            result = env.step(action)
            if result.terminated:
                terminated = True
                break
        
        assert terminated
        assert result.info.get("off_track", False)
    
    def test_episode_truncates_at_max_steps(self, env):
        """Episode should truncate at max steps."""
        env = StubRacingEnv(max_steps=10)
        env.reset()
        
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        
        for i in range(15):
            result = env.step(action)
            if result.truncated:
                break
        
        assert result.truncated
    
    def test_progress_increases_with_throttle(self, env):
        """Track position should increase with throttle."""
        env.reset()
        
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        initial_progress = 0.0
        for _ in range(100):
            result = env.step(action)
        
        assert result.info["lap_progress"] > initial_progress
    
    def test_velocity_increases_with_throttle(self, env):
        """Velocity should increase with throttle."""
        obs, _ = env.reset()
        initial_velocity = obs[0]
        
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        for _ in range(10):
            result = env.step(action)
        
        assert result.info["velocity"] > initial_velocity * 100  # Denormalized


class TestGymWrapper:
    
    @pytest.fixture
    def wrapped_env(self):
        env = StubRacingEnv()
        return GymWrapper(env)
    
    def test_observation_space_type(self, wrapped_env):
        """Observation space should be Box."""
        from gymnasium.spaces import Box
        assert isinstance(wrapped_env.observation_space, Box)
    
    def test_action_space_type(self, wrapped_env):
        """Action space should be Box."""
        from gymnasium.spaces import Box
        assert isinstance(wrapped_env.action_space, Box)
    
    def test_reset_gymnasium_api(self, wrapped_env):
        """Reset should follow Gymnasium API."""
        obs, info = wrapped_env.reset()
        assert obs.shape == wrapped_env.observation_space.shape
        assert isinstance(info, dict)
    
    def test_step_gymnasium_api(self, wrapped_env):
        """Step should follow Gymnasium API."""
        wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        assert obs.shape == wrapped_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestMakeEnv:
    
    def test_make_stub_env(self, config):
        """make_env should create stub environment."""
        env = make_env(config)
        assert isinstance(env, StubRacingEnv)
    
    def test_make_env_unknown_raises(self, config):
        """make_env should raise for unknown environment."""
        config["env"]["name"] = "unknown"
        with pytest.raises(ValueError):
            make_env(config)
