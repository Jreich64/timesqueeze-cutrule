from timesqueeze.data.orcl import OrclData, load_orcl


def small_orcl_data(days=5):
  """The first `days` ORCL days split 3 / 1 / 1, for fast end-to-end tests."""
  d = load_orcl()
  n = int(d.day_starts[days])
  starts = d.day_starts[:days]
  return OrclData(d.returns[:n], d.dt[:n], starts, int(starts[3]), int(starts[4]), d.time_unit)
