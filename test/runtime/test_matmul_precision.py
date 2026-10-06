import unittest
from tinygrad import Context, Tensor, dtypes


class TestMatmulPrecision(unittest.TestCase):
  def test_fallback_products(self):
    cases = ((dtypes.fp8e4m3, 96.0), (dtypes.fp8e5m2, 1.25), (dtypes.half, 256.0),
             (dtypes.half, 1+2**-10), (dtypes.bfloat16, 1+2**-7))
    for dt, value in cases:
      for size, tc in ((1, 1), (32, 0)):
        with self.subTest(dtype=dt, size=size, tc=tc), Context(TC=tc):
          a = Tensor([[value]*size for _ in range(size)], dtype=dt).realize()
          result = a.matmul(a, dtype=dtypes.float32)
          self.assertEqual(result.dtype, dtypes.float32)
          self.assertEqual(result.tolist(), [[size*value*value]*size for _ in range(size)])

  @Context(TC=0)
  def test_default_dtype(self):
    for dt, value in ((dtypes.fp8e4m3, 1.25), (dtypes.fp8e5m2, 1.25), (dtypes.half, 1+2**-10), (dtypes.bfloat16, 1+2**-7)):
      with self.subTest(dtype=dt):
        # Cancellation makes product rounding visible even after the final cast to the input dtype.
        a = Tensor([[value, value]], dtype=dt).realize()
        b = Tensor([[value], [-1]], dtype=dt).realize()
        result = a @ b
        self.assertEqual(result.dtype, dt)
        self.assertEqual(result.realize().float().item(), value*(value-1))

  @Context(TC=0)
  def test_integer_products(self):
    a = Tensor([[100]], dtype=dtypes.int8).realize()
    self.assertEqual(a.matmul(a, dtype=dtypes.int32).item(), (a*a).cast(dtypes.int32).item())

  @Context(TC=0)
  def test_backward(self):
    for dt in (dtypes.half, dtypes.bfloat16):
      with self.subTest(dtype=dt):
        a = Tensor([[1, 2], [3, 4]], dtype=dt).realize()
        b = Tensor([[5, 6], [7, 8]], dtype=dt).realize()
        da, db = a.matmul(b, dtype=dtypes.float32).sum().gradient(a, b)
        self.assertEqual(da.dtype, dt)
        self.assertEqual(db.dtype, dt)
        self.assertEqual(da.float().tolist(), [[11, 15], [11, 15]])
        self.assertEqual(db.float().tolist(), [[4, 4], [6, 6]])


if __name__ == "__main__": unittest.main()
