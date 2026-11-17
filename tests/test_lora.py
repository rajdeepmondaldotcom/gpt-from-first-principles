import torch

from gpt.gpt2 import loss_on
from gpt.lora import GRIDS, LoRA, add_lora, merge
from gpt.model import GPT
from gpt.optim import AdamW


def model():
    torch.manual_seed(0)
    return GPT(vocab=65, context=16, width=32, heads=4, layers=2, kv_heads=2)


def train(net, steps=20):
    torch.manual_seed(1)
    ids = torch.randint(65, (8, 17))
    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], rate=1e-2)
    for _ in range(steps):
        net.zero_grad()
        loss = loss_on(net, ids[:, :-1], ids[:, 1:])
        loss.backward()
        optimizer.step()
    return loss.item()


def test_adding_lora_changes_nothing_until_it_is_trained():
    net, ids = model(), torch.randint(65, (2, 16))
    before = net(ids)
    add_lora(net, rank=4)
    assert torch.equal(net(ids), before)


def test_only_the_corrections_train_and_they_are_small():
    net = model()
    whole = sum(p.numel() for p in net.parameters())
    add_lora(net, rank=4)
    trained = [name for name, p in net.named_parameters() if p.requires_grad]
    assert all(name.endswith((".down", ".up")) for name in trained) and len(trained) == 8
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert count == 2 * (4 * (32 + 32) + 4 * (32 + 16))  # per block: q is 32 by 32, v is 16 by 32
    assert count < whole / 10


def test_training_moves_the_corrections_and_not_one_frozen_weight():
    net = model()
    frozen = {name: p.clone() for name, p in net.named_parameters()}
    add_lora(net, rank=4)
    first = train(net, steps=1)
    assert train(net) < first - 0.1  # it learns
    now = dict(net.named_parameters())
    for name, old in frozen.items():  # q and v moved one level down, inside their LoRA
        name = name.replace("attn.q.", "attn.q.layer.").replace("attn.v.", "attn.v.layer.")
        assert torch.equal(now[name], old), name
    assert net.blocks[0].attn.q.up.abs().sum() > 0


def test_merged_equals_unmerged_and_leaves_a_plain_model():
    net, ids = model(), torch.randint(65, (2, 16))
    names = [name for name, _ in net.named_parameters()]
    add_lora(net, rank=4)
    train(net)
    with torch.no_grad():
        unmerged = net(ids)
        merge(net)
        assert torch.allclose(net(ids), unmerged, atol=1e-5, rtol=0)
    assert [name for name, _ in net.named_parameters()] == names
    assert not any(isinstance(part, LoRA) for part in net.modules())


def test_a_correction_can_sit_beside_each_grid_of_a_block():
    net, ids = model(), torch.randint(65, (2, 16))
    names, before = [name for name, _ in net.named_parameters()], net(ids)
    add_lora(net, rank=4, names=GRIDS)
    assert torch.equal(net(ids), before)  # still no change at the start
    trained = [name for name, p in net.named_parameters() if p.requires_grad]
    assert len(trained) == 2 * 7 * 2  # two blocks, seven grids each, a down and an up for each
    first = train(net, steps=1)
    assert train(net) < first - 0.1  # it learns
    with torch.no_grad():
        unmerged = net(ids)
        merge(net)
        assert torch.allclose(net(ids), unmerged, atol=1e-5, rtol=0)
    assert [name for name, _ in net.named_parameters()] == names  # a plain model again


def test_the_change_a_trained_correction_makes_has_its_rank():
    net = model()
    old = net.blocks[0].attn.q.weight.clone()
    add_lora(net, rank=3)
    train(net)
    one = net.blocks[0].attn.q
    change = (one.scale * one.up @ one.down).detach()
    assert torch.linalg.matrix_rank(change).item() == 3  # a 32 by 32 grid could have rank 32
    merge(net)
    assert torch.allclose(net.blocks[0].attn.q.weight - old, change, atol=1e-6)
