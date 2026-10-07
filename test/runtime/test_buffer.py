import unittest, struct, pickle, weakref, gc
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine.jit import JitError
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp

class TestBuffer(unittest.TestCase):
  def test_typed_interpretation(self):
    b = Buffer("CPU", 8, initial_value=struct.pack("ff", 1, 2))
    floats, ints = (UOp.from_buffer(b, dt) for dt in (dtypes.float32, dtypes.uint32))
    self.assertIs(floats.buffer, ints.buffer)
    self.assertIs(floats.storage_base, ints.storage_base)
    self.assertEqual((floats.shape, ints.shape), ((2,), (2,)))
    self.assertEqual((Tensor(floats) + 1).tolist(), [2, 3])
    self.assertEqual(Tensor(ints).tolist(), [0x3f800000, 0x40000000])
    self.assertEqual(b.numpy(dtypes.float32).tolist(), [1, 2])

  def test_overlapping_interpretations(self):
    for dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8):
      for reverse in (False, True):
        with self.subTest(dtype=dtype, reverse=reverse):
          b = Buffer("CPU", 4096, initial_value=struct.pack("1024f", *range(1, 1025)))
          types = (dtype, dtypes.float32) if reverse else (dtypes.float32, dtype)
          a, c = [Tensor(UOp.from_buffer(b, dt)) for dt in types]
          f, i = (c, a) if reverse else (a, c)
          f.assign(i.bitcast(dtypes.float32).flip(0)).realize()
          self.assertEqual(f.tolist(), list(range(1024, 0, -1)))

  def test_rewrap_new_buffer(self):
    f = UOp.new_buffer("CPU", 1024, dtypes.float32)
    f.buffer.ensure_allocated().copy_from(Buffer("CPU", 4096, initial_value=struct.pack("1024f", *range(1, 1025))))
    i = UOp.from_buffer(f.buffer, dtypes.uint32)
    self.assertIs(i.storage_base, f)
    out = Tensor(f).assign(Tensor(i).flip(0).bitcast(dtypes.float32)).realize()
    self.assertEqual(out.tolist(), list(range(1024, 0, -1)))

  def test_overlapping_buffer_views(self):
    b = Buffer("CPU", 4100, initial_value=struct.pack("1025f", *range(1025)))
    dst = Tensor(UOp.from_buffer(b.view(4096, 4), dtypes.float32))
    src = Tensor(UOp.from_buffer(b.view(4096, 0), dtypes.uint16))
    self.assertIs(dst.uop.storage_base, src.uop.storage_base)
    dst.assign(src.bitcast(dtypes.float32)).realize()
    self.assertEqual(b.numpy(dtypes.float32).tolist(), [0] + list(range(1024)))

  def test_jit_aliased_interpretations(self):
    for dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8):
      with self.subTest(dtype=dtype):
        @TinyJit
        def add(f, i): return (f + i.bitcast(dtypes.float32)).realize()
        b = Buffer("CPU", 16, initial_value=struct.pack("4f", 1, 2, 3, 4))
        f, i = (Tensor(UOp.from_buffer(b, dt)) for dt in (dtypes.float32, dtype))
        with self.assertRaisesRegex(JitError, "duplicate inputs"): add(f, i)
        for step in range(4):
          data = struct.pack("4f", *range(step, step+4))
          f, i = (Tensor(UOp.from_buffer(Buffer("CPU", 16, initial_value=data), dt)) for dt in (dtypes.float32, dtype))
          self.assertEqual(add(f, i).tolist(), [2*x for x in range(step, step+4)])
        alias = Tensor(UOp.from_buffer(f.uop.buffer, dtype))
        with self.assertRaisesRegex(JitError, "duplicate inputs"): add(f, alias)

  def test_jit_typed_view(self):
    @TinyJit
    def add(x): return (x + 1).realize()
    for step in range(4):
      f = UOp.new_buffer("CPU", 4, dtypes.float32)
      f.buffer.ensure_allocated().copy_from(Buffer("CPU", 16, initial_value=struct.pack("4I", *range(step, step+4))))
      self.assertEqual(add(Tensor(UOp.from_buffer(f.buffer, dtypes.uint32))).tolist(), list(range(step+1, step+5)))

  def test_interpretation_lifetime(self):
    f = UOp.new_buffer("CPU", 4, dtypes.float32)
    i = UOp.from_buffer(f.buffer, dtypes.uint16)
    buffer_ref, root_ref = weakref.ref(f.buffer), weakref.ref(f)
    del f
    self.assertIs(i.storage_base, root_ref())
    del i
    gc.collect()
    self.assertIsNone(root_ref())
    self.assertIsNone(buffer_ref())

  def test_pickle_shared_interpretations(self):
    b = Buffer("CPU", 16, initial_value=struct.pack("4f", 1, 2, 3, 4))
    f = UOp.from_buffer(b, dtypes.float32)
    f, i = pickle.loads(pickle.dumps((f, UOp.from_buffer(b, dtypes.uint16))))
    self.assertIs(f.storage_base, i.storage_base)
    self.assertIs(UOp.from_buffer(f.buffer, dtypes.uint8).storage_base, f)
    Tensor(f).assign(Tensor(i).bitcast(dtypes.float32).flip(0)).realize()
    self.assertEqual(f.buffer.numpy(dtypes.float32).tolist(), [4, 3, 2, 1])

  def test_invalid_interpretation(self):
    for dt in (dtypes.uint32, dtypes.void, dtypes.weakint, dtypes.weakfloat):
      with self.subTest(dtype=dt), self.assertRaises(AssertionError): UOp.from_buffer(Buffer("CPU", 3), dt)

  def test_byte_view_write(self):
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.float32):
      for offset in (1, 4):
        for overlap in (False, True):
          with self.subTest(dtype=dtype, offset=offset, overlap=overlap):
            b = Buffer("CPU", 16, initial_value=bytes(range(16)))
            root = UOp.from_buffer(b, dtype)
            view = Tensor(UOp.from_buffer(b.view(8, offset), dtypes.uint8))
            src = Tensor(UOp.from_buffer(b.view(8, 0), dtypes.uint8)) if overlap else Tensor([42]*8, device="CPU", dtype=dtypes.uint8)
            view.assign(src).realize()
            self.assertEqual(list(b.as_memoryview()), list(range(offset)) + (list(range(8)) if overlap else [42]*8) + list(range(offset+8, 16)))
            self.assertIs(root.buffer, b)

  def test_unaligned_typed_view_write(self):
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.float32):
      for dst_offset, src_offset in ((1, 2), (2, 1)):
        with self.subTest(dtype=dtype, dst_offset=dst_offset):
          b = Buffer("CPU", 16, initial_value=bytes(range(16)))
          root = UOp.from_buffer(b, dtype)
          dst = Tensor(UOp.from_buffer(b.view(8, dst_offset), dtypes.uint32))
          src = Tensor(UOp.from_buffer(b.view(8, src_offset), dtypes.uint32))
          dst.assign(src).realize()
          expected = bytearray(range(16))
          expected[dst_offset:dst_offset+8] = bytes(range(src_offset, src_offset+8))
          self.assertEqual(bytes(b.as_memoryview()), expected)
          self.assertIs(root.buffer, b)

  def test_multibuffer_interpretations(self):
    for mode in ("aggregate", "stacked", "mixed", "raw"):
      for original in ("source", "target", "neither"):
        with self.subTest(mode=mode, original=original):
          if mode == "aggregate": m = UOp.new_buffer(("CPU", "CPU:1"), 1024, dtypes.float32)
          elif mode == "raw": m = UOp.from_buffer(MultiBuffer(("CPU", "CPU:1"), 4096), dtypes.float32)
          else:
            first = UOp.new_buffer("CPU", 1024, dtypes.uint32 if mode == "mixed" else dtypes.float32)
            m = first.bitcast(dtypes.float32).mstack(UOp.new_buffer("CPU:1", 1024, dtypes.float32))
          for b in m.buffer.bufs:
            b.ensure_allocated().copy_from(Buffer("CPU", 4096, initial_value=struct.pack("1024f", *range(1, 1025))))
          f, i = (Tensor(UOp.from_buffer(m.buffer, dt)) for dt in (dtypes.float32, dtypes.uint32))
          self.assertIs(f.uop.storage_base, i.uop.storage_base)
          @TinyJit
          def add(a, b): return (a + b.bitcast(dtypes.float32)).realize()
          with self.assertRaisesRegex(JitError, "duplicate inputs"): add(f, i)
          with self.assertRaisesRegex(JitError, "duplicate inputs"): add(Tensor(m), i)
          dst = Tensor(m) if original == "target" else f
          src = Tensor(m) if original == "source" else i.bitcast(dtypes.float32)
          dst.assign(src.flip(0)).realize()
          self.assertEqual(dst.tolist(), list(range(1024, 0, -1)))
          for b in m.buffer.bufs: self.assertEqual(b.numpy(dtypes.float32).tolist(), list(range(1024, 0, -1)))

  def test_view_bounds(self):
    b = Buffer("CPU", 16)
    v = b.view(8, 4)
    for nbytes, offset in ((9, 0), (4, 5), (1, -1), (-1, 0)):
      with self.subTest(nbytes=nbytes, offset=offset), self.assertRaises(AssertionError): v.view(nbytes, offset)
    self.assertEqual((v.view(0, 8).nbytes, v.view(0, 8).offset), (0, 12))

  def test_copy_typed_view(self):
    src = Buffer("CPU", 12, initial_value=struct.pack("fff", 1, 2, 3))
    dst = Buffer("CPU", 8, preallocate=True)
    dst.copy_from(src.view(8, 4).ensure_allocated())
    self.assertEqual(Tensor(UOp.from_buffer(dst, dtypes.float32)).tolist(), [2, 3])

  def test_pickle_typed_view(self):
    for device in ("CPU", "NPY"):
      for protocol in (4, 5):
        with self.subTest(device=device, protocol=protocol):
          b = Buffer(device, 12, initial_value=struct.pack("fff", 1, 2, 3))
          u = UOp.from_buffer(b.view(8, 4).ensure_allocated(), dtypes.float32)
          v = pickle.loads(pickle.dumps(u, protocol=protocol))
          self.assertEqual((v.dtype, v.buffer.nbytes, v.buffer.offset), (dtypes.float32, 8, 4))
          self.assertEqual(Tensor(v).tolist(), [2, 3])

  def test_byte_limit(self):
    with Context(MAX_BUFFER_SIZE=8):
      self.assertEqual(Buffer("CPU", 8).ensure_allocated().nbytes, 8)
      with self.assertRaises(RuntimeError): Buffer("CPU", 9).ensure_allocated()

  def test_host_view(self):
    b = Buffer("CPU", 16)
    v = b.view(4, 4)
    host = v.host
    host.view(fmt='H')[0] = 0x1234
    self.assertEqual(b.host.view(fmt='H')[2], 0x1234)
    self.assertEqual(v._buf, b._buf + 4)
    self.assertIs(v.host, host)
    self.assertIs(v.meta, b.meta)

  def test_mapping(self):
    b = Buffer("CPU", 8, initial_value=b"abcdefgh")
    self.assertEqual(b.get_buf("PYTHON"), b._buf)
    v = b.view(4, 2)
    mapped = v.get_storage("PYTHON")
    self.assertEqual(mapped.buf, b._buf + 2)
    self.assertEqual(bytes(mapped.host.mv), b"cdef")
    self.assertIs(mapped.host, v.host)
    self.assertIsNone(mapped.meta)
    self.assertIs(v.get_storage("PYTHON"), mapped)

  def test_view_reallocation(self):
    b = Buffer("CPU", 8)
    v = b.view(4, 2)
    old = v.get_storage("PYTHON")
    b.deallocate()
    b.allocate()
    self.assertFalse(v.is_allocated())
    v.host[:] = b"test"
    self.assertIsNot(v.get_storage("PYTHON"), old)
    self.assertEqual(bytes(v.get_storage("PYTHON").host.mv), b"test")

  def test_cache_owned_storage_only(self):
    for opaque in (None, memoryview(bytearray(8))):
      with self.subTest(imported=opaque is not None), Context(LRU=1):
        b = Buffer("PYTHON", 8, opaque=opaque)
        buf = b._buf
        b.deallocate()
        self.assertEqual(b._buf is buf, opaque is None)

if __name__ == "__main__": unittest.main()
