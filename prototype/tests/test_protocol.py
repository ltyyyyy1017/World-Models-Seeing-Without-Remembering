import copy
import numpy as np
from privacywm.env import World, identity_sequence


def test_identity_does_not_change_physics_or_oracle():
    for seed in range(12):
        world = World(seed)
        altered = copy.deepcopy(world)
        altered.identity = 29
        for _ in range(60):
            action = world.oracle()
            assert altered.oracle() == action
            assert world.step(action) == altered.step(action)
            assert world.agent == altered.agent
            assert world.carrying == altered.carrying
            if world.success:
                break
        assert world.success


def test_appearance_changes_pixels_but_not_other_metadata():
    world = World(2)
    world.agent, world.person, world.heading = (3, 4), (4, 4), 1
    before = world.metadata()
    a, va = world.render(identity=0)
    b, vb = world.render(identity=29)
    assert va and vb and not np.array_equal(a, b)
    assert before == world.metadata()
    assert a.shape == (64, 64, 3) and a.dtype == np.uint8
    masked, seen = world.render(mask_person=True)
    assert not seen and not np.array_equal(a, masked)


def test_schedules_and_population_balance():
    for mode in ("S0", "S3"):
        seq = np.array([identity_sequence(mode, 60, ep) for ep in range(30)])
        assert np.unique(np.bincount(seq.ravel(), minlength=30)).size == 1
    periodic = identity_sequence("S3", 120, 4)
    assert np.all(periodic[1:] == (periodic[:-1] + 1) % 30)
    stable = identity_sequence("S0", 120, 4)
    assert np.all(stable == stable[0])
    random = identity_sequence("S2", 30000, 4)
    assert 0.02 < np.mean(random[:-1] == random[1:]) < 0.05


def test_occlusion_by_wall():
    world = World(0)
    world.agent, world.person, world.heading = (3, 2), (5, 2), 1
    _, visible = world.render()
    assert not visible


def test_visibility_selection_is_identity_independent():
    for seed in range(10):
        world = World(seed)
        for _ in range(15):
            assert world.render(identity=0)[1] == world.render(identity=29)[1]
            world.step(world.oracle())


def test_paired_initialization_and_finite_gradients():
    import pytest
    torch = pytest.importorskip("torch")
    from privacywm.models import WorldModel, loss_terms
    torch.set_num_threads(2)
    torch.manual_seed(3)
    first = WorldModel(8)
    torch.manual_seed(3)
    second = WorldModel(8)
    assert all(torch.equal(a, b) for a, b in zip(first.encoder.parameters(), second.encoder.parameters()))
    x, nxt = torch.rand(8, 3, 64, 64), torch.rand(8, 3, 64, 64)
    for kind in ("A", "B", "C"):
        first.zero_grad(set_to_none=True)
        loss, metrics = loss_terms(first, x, nxt, torch.arange(8) % 5, kind)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in first.encoder.parameters())


def test_oracle_upper_bound():
    successes = 0
    for seed in range(100):
        world = World(500000 + seed)
        for _ in range(80):
            world.step(world.oracle())
            if world.success:
                successes += 1
                break
    assert successes >= 90


def test_dataset_pairing_splits_and_tamper_detection(tmp_path):
    import pytest
    from privacywm.data import collect, load_data
    collect(tmp_path, episodes=2, horizon=4, modes=("S0", "S2"))
    a = load_data(tmp_path, "S0", "world_train")
    b = load_data(tmp_path, "S2", "world_train")
    test = load_data(tmp_path, "S0", "probe_test")
    assert not set(a["episode_ids"]) & set(test["episode_ids"])
    for key in ("actions", "position", "rewards", "done", "visible"):
        assert np.array_equal(a[key], b[key])
    path = tmp_path / "S0_world_train.npz"
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_data(tmp_path, "S0", "world_train")
