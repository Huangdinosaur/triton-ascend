import triton
import triton.language as tl
import torch


@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tmp1 = tl.atomic_xchg(out_ptr0 + xindex, tmp0, xmask)
    tl.store(out_ptr1 + xindex, tmp1, xmask)


def test_atomic_xchg():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    split_size = shape[0] // ncore

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()
    pointer_old = torch.full_like(val, -10).npu()

    # 每个 program 只交换自己那一段，交换后 pointer == val、pointer_old == 原 pointer
    pointer_ref = val.clone()
    pointer_old_ref = pointer.clone()

    n_elements = shape[0] * shape[1]
    atomic_xchg[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    assert torch.equal(pointer, pointer_ref)
    assert torch.equal(pointer_old, pointer_old_ref)


if __name__ == "__main__":
    test_atomic_xchg()
