import unittest, struct, pickle
from tinygrad import Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp

class TestBuffer(unittest.TestCase):
  def test_typed_interpretation(self):
    b = Buffer("CPU", 8, initial_value=struct.pack("ff", 1, 2))
    floats, ints = (UOp.from_buffer(b, dt) for dt in (dtypes.float32, dtypes.uint32))
    self.assertIs(floats.buffer, ints.buffer)
    self.assertEqual((floats.shape, ints.shape), ((2,), (2,)))
    self.assertEqual((Tensor(floats) + 1).tolist(), [2, 3])
    self.assertEqual(Tensor(ints).tolist(), [0x3f800000, 0x40000000])
    self.assertEqual(b.numpy(dtypes.float32).tolist(), [1, 2])

  def test_invalid_interpretation(self):
    for dt in (dtypes.uint32, dtypes.void, dtypes.weakint, dtypes.weakfloat):
      with self.subTest(dtype=dt), self.assertRaises(AssertionError): UOp.from_buffer(Buffer("CPU", 3), dt)

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
