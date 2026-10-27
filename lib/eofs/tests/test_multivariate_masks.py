"""Masked cells must not contribute to multivariate EOF analysis."""
import numpy as np
import pytest

from eofs.multivariate.standard import MultivariateEof


@pytest.mark.parametrize('both_masked', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_masked_values_do_not_contribute(both_masked, weighted):
    rng = np.random.RandomState(27)
    first = np.ma.array(rng.normal(size=(20, 3)),
                        mask=np.broadcast_to([False, True, False],
                                             (20, 3)).copy())
    second = rng.normal(size=(20, 4))
    if both_masked:
        second = np.ma.array(second,
                             mask=np.broadcast_to([True, False, False, False],
                                                  second.shape).copy())
    first -= first.mean(axis=0)
    second -= second.mean(axis=0)
    weights = [np.array([0.5, 1.0, 0.8]), None] if weighted else None
    fields = [first, second]
    # Changing only hidden values must leave the scientific solution unchanged.
    reference = MultivariateEof(fields, weights=weights)
    expected_eigenvalues = reference.eigenvalues()
    expected_pcs = reference.pcs()
    for field in fields:
        if np.ma.isMaskedArray(field):
            field.data[np.ma.getmaskarray(field)] = np.arange(20) * 1000.0
    result = MultivariateEof(fields, weights=weights)
    np.testing.assert_allclose(result.eigenvalues(), expected_eigenvalues)
    np.testing.assert_allclose(result.pcs(), expected_pcs)
    for field, eof in zip(fields, result.eofs()):
        expected_mask = np.broadcast_to(np.ma.getmaskarray(field)[0], eof.shape)
        np.testing.assert_array_equal(np.ma.getmaskarray(eof), expected_mask)
    projections = result.projectField(fields)
    np.testing.assert_allclose(projections, expected_pcs)


@pytest.mark.parametrize('dtype', [np.int32, np.uint64, np.bool_])
def test_masked_integer_fields(dtype):
    values = np.arange(60).reshape(20, 3) % 7
    if dtype == np.bool_:
        values = values > 3
    first = np.ma.array(values.astype(dtype),
                        mask=np.broadcast_to([False, True, False],
                                             values.shape).copy())
    second = (values[:, :2] % 2).astype(dtype)
    solver = MultivariateEof([first, second])
    reference = MultivariateEof([first.astype(float), second.astype(float)])
    np.testing.assert_allclose(solver.eigenvalues(), reference.eigenvalues())
    np.testing.assert_array_equal(np.ma.getmaskarray(solver.eofs()[0]),
                                  np.ma.getmaskarray(reference.eofs()[0]))
