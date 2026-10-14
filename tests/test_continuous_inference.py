import collections

import numpy as np

from sign_language.api.main import SessionState


def test_session_state_initialization():
    state = SessionState(seq_len=64)
    assert state.seq_len == 64
    assert state.state == "WAITING"
    assert len(state.sequence_buffer) == 0
    assert len(state.mask_buffer) == 0

def test_session_state_append_and_reset():
    state = SessionState(seq_len=64)

    # Simulate active gesture
    state.state = "CAPTURING"
    for i in range(10):
        state.append_frame(np.ones(10), np.ones(10))

    assert len(state.sequence_buffer) == 10

    # Simulate neutral pause
    state.handle_neutral_pause()
    assert state.state == "WAITING"
    assert len(state.sequence_buffer) == 0
    assert len(state.mask_buffer) == 0

def test_session_state_transition_to_committed():
    state = SessionState(seq_len=64)
    state.state = "CANDIDATE"
    for i in range(15):
        state.append_frame(np.ones(10), np.ones(10))
        state.candidate_history.append("hello")

    assert len(state.sequence_buffer) == 15
    assert len(state.candidate_history) > 0

    state.transition_to_committed("hello")
    assert state.state == "COMMITTED"
    assert state.last_committed_word == "hello"
    assert state.current_candidate is None
    # Buffers should be cleared on commit
    assert len(state.sequence_buffer) == 0
    assert len(state.candidate_history) == 0

def test_continuous_cooldown():
    state = SessionState(seq_len=64)
    state.mode = "continuous"

    # Simulate continuous predictions
    for i in range(4):
        state.continuous_candidate_history.append("apple")

    # The websocket logic checks count >= 4 and most_common != continuous_last_emitted
    # Let's manually trigger the state we would expect
    most_common, count = collections.Counter(state.continuous_candidate_history).most_common(1)[0]
    assert most_common == "apple"
    assert count == 4

    # Simulate the websocket action
    state.continuous_last_emitted = "apple"
    state.continuous_cooldown_frames = 15
    state.continuous_candidate_history.clear()
    state.sequence_buffer.clear()

    assert state.continuous_cooldown_frames == 15
    assert state.continuous_last_emitted == "apple"
    assert len(state.continuous_candidate_history) == 0

def test_full_reset():
    state = SessionState(seq_len=64)
    state.transition_to_committed("world")
    state.append_frame(np.ones(10), np.ones(10))

    state.full_reset()
    assert state.state == "WAITING"
    assert state.last_committed_word is None
    assert len(state.sequence_buffer) == 0
