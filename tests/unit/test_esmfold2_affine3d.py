"""Quaternion, rotation-matrix, and rigid-frame algebra of the ESMFold2 structure utilities.

Every check is an identity the mathematics fixes, on CPU: a rotation about an axis moves a known point to a known place,
a transform composed with its inverse is the identity, and the quaternion and matrix forms agree.

Shapes: `b` frames in a batch, `n` points per frame.
"""

import math
import pytest
import torch

from fastplms.models.esmfold2 import esmfold2_affine3d as affine3d
from fastplms.models.esmfold2.esmfold2_affine3d import Affine3D, RotationMatrix, RotationQuat


QUARTER_TURN = math.sqrt(0.5)
Z_QUARTER_TURN = (QUARTER_TURN, 0.0, 0.0, QUARTER_TURN)  # real-first quaternion of a 90 degree turn about z
X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)


def z_quarter_turns(batch: int) -> RotationQuat:
    return RotationQuat(torch.tensor([Z_QUARTER_TURN] * batch))  # (b, 4)


def assert_rotations_agree(first: torch.Tensor, second: torch.Tensor) -> None:
    # first, second: (..., 3, 3)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=0)


def test_the_square_root_has_a_zero_subgradient_where_it_is_not_defined():
    values = torch.tensor([4.0, 0.0, -1.0], requires_grad=True)  # (3,)

    roots = affine3d._sqrt_subgradient(values)  # (3,)
    roots.sum().backward()

    assert roots.tolist() == [2.0, 0.0, 0.0]
    assert values.grad.tolist() == [0.25, 0.0, 0.0]


def test_a_quaternion_times_its_inverse_is_the_identity_and_a_quarter_turn_moves_x_to_y():
    quaternion = torch.tensor([Z_QUARTER_TURN])  # (1, 4)

    product = affine3d._quat_mult(quaternion, affine3d._quat_invert(quaternion))  # (1, 4)
    moved = affine3d._quat_rotation(quaternion, torch.tensor([X_AXIS]))  # (1, 3)

    torch.testing.assert_close(product, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(moved, torch.tensor([Y_AXIS]), atol=1e-6, rtol=0)


def test_the_graham_schmidt_frame_is_orthonormal_right_handed_and_points_along_x():
    generator = torch.Generator().manual_seed(0)
    x_axis = torch.randn(5, 3, generator=generator)  # (b, 3)
    xy_plane = torch.randn(5, 3, generator=generator)  # (b, 3)

    frame = affine3d._graham_schmidt(x_axis, xy_plane)  # (b, 3, 3), columns are the axes

    torch.testing.assert_close(frame.transpose(-1, -2) @ frame, torch.eye(3).expand(5, 3, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.linalg.det(frame), torch.ones(5), atol=1e-5, rtol=0)
    torch.testing.assert_close(frame[..., 0], torch.nn.functional.normalize(x_axis, dim=-1), atol=1e-5, rtol=0)


def test_a_quaternion_rotation_reports_its_tensor_shape_dtype_and_device():
    rotation = RotationQuat.identity((2, 3), dtype=torch.float32)

    assert rotation.shape == (2, 3)
    assert rotation.tensor.shape == (2, 3, 4)
    assert rotation.dtype == torch.float32
    assert rotation.device == torch.device("cpu")
    assert rotation.requires_grad is False
    assert rotation.tensor[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_a_quaternion_rotation_rejects_input_that_is_not_a_quaternion():
    with pytest.raises(TypeError, match="Torch tensor"):
        RotationQuat([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="trailing dimension 4"):
        RotationQuat(torch.ones(3))
    with pytest.raises(TypeError, match="boolean"):
        RotationQuat(torch.ones(4), normalized=1)


def test_a_normalized_quaternion_has_unit_length_and_a_nonnegative_real_part():
    raw = torch.tensor([[-2.0, 0.0, 0.0, 0.0]])  # (1, 4)

    normalized = RotationQuat(raw, normalized=True)
    same = normalized.normalized()
    from_raw = RotationQuat(raw).normalized()

    assert normalized.tensor.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert same is normalized
    assert from_raw.tensor.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert normalized.as_quat() is normalized


def test_indexing_a_rotation_keeps_the_component_axis():
    quaternions = RotationQuat.random((4, 2))
    matrices = RotationMatrix.random((4, 2))

    assert quaternions[1].shape == (2,)
    assert quaternions[1:3, 0].shape == (2,)
    assert torch.equal(quaternions[1].tensor, quaternions.tensor[1])
    assert matrices[1].shape == (2,)
    assert matrices[1:3, 0].shape == (2,)
    assert torch.equal(matrices[1].tensor, matrices.tensor[1])


def test_a_quarter_turn_quaternion_converts_to_the_matrix_and_back():
    quaternions = z_quarter_turns(2)

    matrix = quaternions.as_matrix()  # (2, 3, 3)
    back = matrix.as_quat()

    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).expand(2, 3, 3)  # (2, 3, 3)
    assert_rotations_agree(matrix.to_3x3(), expected)
    assert matrix.as_matrix() is matrix
    torch.testing.assert_close(back.tensor, quaternions.tensor, atol=1e-5, rtol=0)


def test_random_rotations_are_orthonormal_and_survive_the_round_trip_through_a_quaternion():
    torch.manual_seed(0)

    from_quaternion = RotationQuat.random((6,))
    from_matrix = RotationMatrix.random((6,))
    round_trip = from_matrix.as_quat().as_matrix()  # (6, 3, 3)

    for rotation in (from_quaternion.as_matrix(), from_matrix):
        matrices = rotation.to_3x3()  # (6, 3, 3)
        torch.testing.assert_close(matrices @ matrices.transpose(-1, -2), torch.eye(3).expand(6, 3, 3), atol=1e-5, rtol=0)
    assert_rotations_agree(round_trip.to_3x3(), from_matrix.to_3x3())


def test_the_matrix_rotation_accepts_nine_values_and_refuses_other_shapes():
    flat = torch.eye(3).reshape(1, 9)  # (1, 9)

    assert RotationMatrix(flat).to_3x3().shape == (1, 3, 3)
    assert RotationMatrix(flat).tensor.shape == (1, 9)
    with pytest.raises(TypeError, match="Torch tensor"):
        RotationMatrix([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="trailing shape"):
        RotationMatrix(torch.ones(2, 4))


def test_the_identity_rotation_leaves_points_alone_in_both_forms():
    points = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])  # (n, 3)

    for rotation_type in (RotationQuat, RotationMatrix):
        identity = rotation_type.identity((2,))

        torch.testing.assert_close(identity.apply(points), points, atol=1e-6, rtol=0)
        assert identity.shape == (2,)


def test_a_rotation_applies_the_same_way_as_a_quaternion_and_as_a_matrix_with_or_without_a_batch_axis():
    quaternions = z_quarter_turns(1)  # (1,)
    points = torch.tensor([X_AXIS, Y_AXIS])  # (n, 3)
    expected = torch.tensor([Y_AXIS, [-1.0, 0.0, 0.0]])  # (n, 3)

    by_quaternion = quaternions.apply(points)  # (n, 3)
    by_shared_matrix = quaternions.as_matrix().apply(points)  # (n, 3)
    by_batched_matrix = z_quarter_turns(2).as_matrix().apply(points)  # (n, 3)

    torch.testing.assert_close(by_quaternion, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(by_shared_matrix, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(by_batched_matrix, expected, atol=1e-6, rtol=0)


def test_two_quarter_turns_compose_to_a_half_turn_and_the_inverse_undoes_each():
    quaternions = z_quarter_turns(1)
    matrices = quaternions.as_matrix()
    points = torch.tensor([X_AXIS])  # (1, 3)
    half_turn_of_x = torch.tensor([[-1.0, 0.0, 0.0]])  # (1, 3)

    twice_quaternion = quaternions.compose(quaternions)
    twice_matrix = matrices.compose(matrices)
    mixed_matrix = matrices.convert_compose(quaternions)
    mixed_quaternion = quaternions.convert_compose(matrices)

    for composed in (twice_quaternion, twice_matrix, mixed_matrix, mixed_quaternion):
        torch.testing.assert_close(composed.apply(points), half_turn_of_x, atol=1e-6, rtol=0)
    for rotation in (quaternions, matrices):
        torch.testing.assert_close(rotation.invert().apply(rotation.apply(points)), points, atol=1e-6, rtol=0)


def test_a_rotation_converts_dtype_detaches_and_maps_a_function_over_its_components():
    rotation = RotationMatrix.random((3,))
    tracked = RotationQuat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True))  # (1, 4)

    converted = rotation.to(dtype=torch.float64)
    detached = tracked.detach()
    doubled = tracked.tensor_apply(lambda component: component * 2)

    assert RotationQuat._from_tensor(tracked.tensor).shape == (1,)
    assert tracked.requires_grad and not detached.requires_grad
    assert converted.tensor.dtype == torch.float32  # the matrix form always keeps FP32
    assert doubled.tensor.tolist() == [[2.0, 0.0, 0.0, 0.0]]


def test_a_graham_schmidt_rotation_builds_the_requested_frame():
    x_axis = torch.tensor([[0.0, 2.0, 0.0]])  # (1, 3)
    xy_plane = torch.tensor([[-1.0, 1.0, 0.0]])  # (1, 3)

    rotation = RotationMatrix.from_graham_schmidt(x_axis, xy_plane)  # (1, 3, 3)

    expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])  # (1, 3, 3)
    assert_rotations_agree(rotation.to_3x3(), expected)


def test_an_affine_checks_its_translation_and_rotation_before_it_exists():
    rotation = RotationMatrix.identity((2,))

    with pytest.raises(TypeError, match="trans must be a Torch tensor"):
        Affine3D(trans=[0.0, 0.0, 0.0], rot=rotation)
    with pytest.raises(TypeError, match="Rotation interface"):
        Affine3D(trans=torch.zeros(2, 3), rot=torch.eye(3))
    with pytest.raises(ValueError, match="trailing dimension 3"):
        Affine3D(trans=torch.zeros(2, 4), rot=rotation)
    with pytest.raises(ValueError, match="batch shapes must match"):
        Affine3D(trans=torch.zeros(3, 3), rot=rotation)


def test_an_affine_reports_its_shape_dtype_device_and_stacked_tensor():
    torch.manual_seed(0)
    quaternion_form = Affine3D.random((2, 3), rotation_type=RotationQuat)
    matrix_form = Affine3D.random((2, 3), std=0.0)

    assert quaternion_form.shape == (2, 3)
    assert quaternion_form.dtype == torch.float32
    assert quaternion_form.device == torch.device("cpu")
    assert quaternion_form.requires_grad is False
    assert quaternion_form.tensor.shape == (2, 3, 7)
    assert matrix_form.tensor.shape == (2, 3, 12)
    assert matrix_form.trans.abs().max() == 0


def test_an_affine_rebuilds_from_every_tensor_layout():
    torch.manual_seed(0)
    original = Affine3D.random((4,), rotation_type=RotationQuat)

    from_seven = Affine3D.from_tensor(original.tensor)  # 4 quaternion values and 3 translation values
    from_twelve = Affine3D.from_tensor(original.as_matrix().tensor)  # 9 rotation values and 3 translation values
    from_four_by_four = Affine3D.from_tensor(torch.eye(4).expand(4, 4, 4))
    from_three_by_four = Affine3D.from_tensor(torch.eye(4)[:3].expand(4, 3, 4))
    from_six = Affine3D.from_tensor(torch.zeros(4, 6))
    pair = Affine3D.from_tensor_pair(torch.zeros(4, 3), torch.eye(3).expand(4, 3, 3))

    assert_rotations_agree(from_seven.as_matrix().rot.to_3x3(), original.as_matrix().rot.to_3x3())
    assert_rotations_agree(from_twelve.rot.to_3x3(), original.as_matrix().rot.to_3x3())
    assert torch.equal(from_seven.trans, original.trans)
    for identity in (from_four_by_four, from_three_by_four, pair):
        assert identity.shape == (4,)
        assert_rotations_agree(identity.rot.to_3x3(), torch.eye(3).expand(4, 3, 3))
    assert from_six.rot.tensor.shape == (4, 4)


def test_an_affine_refuses_tensors_of_no_known_layout():
    with pytest.raises(TypeError, match="Torch tensor"):
        Affine3D.from_tensor([0.0] * 7)
    with pytest.raises(ValueError, match="at least one dimension"):
        Affine3D.from_tensor(torch.tensor(1.0))
    with pytest.raises(ValueError, match=r"\(3, 4\) or"):
        Affine3D.from_tensor(torch.zeros(2, 5, 4))
    with pytest.raises(RuntimeError, match="Cannot detect rotation format"):
        Affine3D.from_tensor(torch.zeros(2, 8))


def test_the_identity_affine_copies_the_form_of_an_affine_it_is_given():
    template = Affine3D.random((3,), rotation_type=RotationQuat, dtype=torch.float32)

    plain = Affine3D.identity((3,))
    like = Affine3D.identity(template)

    assert isinstance(plain.rot, RotationMatrix)
    assert isinstance(like.rot, RotationQuat)
    assert like.shape == (3,) and like.dtype == template.dtype
    assert like.trans.abs().max() == 0


def test_an_affine_built_from_three_points_has_its_origin_and_axes_where_the_points_say():
    neg_x_axis = torch.tensor([[-1.0, 0.0, 0.0]])  # (1, 3)
    origin = torch.zeros(1, 3)  # (1, 3)
    xy_plane = torch.tensor([[0.0, 1.0, 0.0]])  # (1, 3)

    frame = Affine3D.from_graham_schmidt(neg_x_axis, origin, xy_plane)  # one frame

    assert_rotations_agree(frame.rot.to_3x3(), torch.eye(3).expand(1, 3, 3))
    assert torch.equal(frame.trans, origin)


def test_affines_concatenate_along_the_batch_axis():
    torch.manual_seed(0)
    first = Affine3D.random((2,))
    second = Affine3D.random((3,))

    joined = Affine3D.cat([first, second])
    last_axis = Affine3D.cat([first[None], second[(slice(0, 2),)][None]], dim=-1)

    assert joined.shape == (5,)
    assert torch.equal(joined.trans[:2], first.trans)
    assert last_axis.shape == (1, 4)
    with pytest.raises(ValueError, match="at least one transform"):
        Affine3D.cat([])
    with pytest.raises(TypeError, match="only Affine3D"):
        Affine3D.cat([first, "second"])


def test_an_affine_indexes_converts_dtype_detaches_and_maps_a_function_over_its_components():
    torch.manual_seed(0)
    affine = Affine3D.random((4,))
    tracked = Affine3D(torch.zeros(1, 3, requires_grad=True), RotationMatrix.identity((1,)))

    picked = affine[1]
    sliced = affine[(slice(1, 3),)]
    doubled = affine.to(dtype=torch.float64)
    detached = tracked.detach()
    shifted = Affine3D.identity((2,)).tensor_apply(lambda component: component + 1)

    assert picked.shape == () and sliced.shape == (2,)
    assert torch.equal(picked.trans, affine.trans[1])
    assert doubled.trans.dtype == torch.float64
    assert tracked.requires_grad and not detached.requires_grad
    assert shifted.trans.tolist() == [[1.0, 1.0, 1.0]] * 2


def test_an_affine_switches_between_matrix_and_quaternion_rotations_without_moving_points():
    torch.manual_seed(0)
    affine = Affine3D.random((3,), rotation_type=RotationQuat)
    points = torch.randn(3, 3)  # (n, 3)

    as_matrix = affine.as_matrix()
    back = as_matrix.as_quat()

    assert isinstance(as_matrix.rot, RotationMatrix) and isinstance(back.rot, RotationQuat)
    torch.testing.assert_close(as_matrix.apply(points), affine.apply(points), atol=1e-5, rtol=0)
    torch.testing.assert_close(back.apply(points), affine.apply(points), atol=1e-5, rtol=0)


def test_composing_an_affine_with_its_inverse_gives_the_identity_and_applying_follows_the_order():
    torch.manual_seed(0)
    first = Affine3D.random((3,))
    second = Affine3D.random((3,))
    points = torch.randn(3, 3)  # (n, 3)

    identity = first.compose(first.invert())
    composed = first.compose(second)

    torch.testing.assert_close(identity.trans, torch.zeros(3, 3), atol=1e-5, rtol=0)
    assert_rotations_agree(identity.rot.to_3x3(), torch.eye(3).expand(3, 3, 3))
    torch.testing.assert_close(composed.apply(points), first.apply(second.apply(points)), atol=1e-5, rtol=0)


def test_composing_across_rotation_forms_needs_autoconvert():
    torch.manual_seed(0)
    matrix_form = Affine3D.random((2,))
    quaternion_form = Affine3D.random((2,), rotation_type=RotationQuat)

    converted = matrix_form.compose(quaternion_form, autoconvert=True)
    rotated_only = matrix_form.compose_rotation(quaternion_form.rot, autoconvert=True)

    assert isinstance(converted.rot, RotationMatrix) and isinstance(rotated_only.rot, RotationMatrix)
    assert torch.equal(rotated_only.trans, matrix_form.trans)
    assert_rotations_agree(rotated_only.rot.to_3x3(), converted.rot.to_3x3())
    same_form = matrix_form.compose_rotation(matrix_form.rot)
    assert_rotations_agree(same_form.rot.to_3x3(), matrix_form.rot.compose(matrix_form.rot).to_3x3())


def test_scaling_and_masking_change_only_the_requested_frames():
    torch.manual_seed(0)
    affine = Affine3D.random((3,))
    masked_out = torch.tensor([True, False, True])  # (3,) frames to reset

    scaled = affine.scale(2.0)
    reset_to_identity = affine.mask(masked_out)
    reset_to_zero = affine.mask(masked_out, with_zero=True)

    assert torch.equal(scaled.trans, affine.trans * 2) and scaled.rot is affine.rot
    assert torch.equal(reset_to_identity.trans[1], affine.trans[1])
    assert reset_to_identity.trans[masked_out].abs().max() == 0
    assert_rotations_agree(reset_to_identity.rot.to_3x3()[0], torch.eye(3))
    assert_rotations_agree(reset_to_identity.rot.to_3x3()[1], affine.rot.to_3x3()[1])
    assert reset_to_zero.tensor[masked_out].abs().max() == 0
    assert torch.equal(reset_to_zero.tensor[1], affine.tensor[1])


def test_residue_frames_sit_on_the_backbone_and_a_missing_residue_takes_the_average_frame():
    backbone = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # N, CA, C of a residue at the origin
            [[10.0, 1.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],  # the same shape 10 angstroms along x
        ]
    )  # (l, 3, 3)
    coords = backbone[None].repeat(2, 1, 1, 1)  # (b, l, 3, 3)
    coords[1, 1] = torch.nan

    frames, present = affine3d.build_affine3d_from_coordinates(coords)  # frames (b, l), mask (b, l)

    assert present.tolist() == [[True, True], [True, False]]
    assert frames.shape == (2, 2)
    torch.testing.assert_close(frames.trans[0, 0], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(frames.trans[0, 1], torch.tensor([10.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    assert torch.isfinite(frames.tensor).all()
    with pytest.raises(TypeError, match="Torch tensor"):
        affine3d.build_affine3d_from_coordinates([[0.0]])
    with pytest.raises(ValueError, match="shape"):
        affine3d.build_affine3d_from_coordinates(torch.zeros(2, 3, 3))
