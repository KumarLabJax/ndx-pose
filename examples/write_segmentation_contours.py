"""
Example script that demonstrates how to write segmentation contours alongside pose estimates.

With one camera, one skeleton, three body parts, and a ContourSeries holding the outline of the
segmented animal on every frame.
"""

import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from ndx_pose import (
    ContourSeries,
    PoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
)

num_frames = 100
frame_rate = 30.0

# initialize an NWBFile object
nwbfile = NWBFile(
    session_description="session_description",
    identifier="identifier",
    session_start_time=datetime.datetime.now(datetime.timezone.utc),
)

# add a subject to the NWB file
subject = Subject(subject_id="subject1", species="Mus musculus")
nwbfile.subject = subject

# create a skeleton that defines the relationship between the markers
skeleton = Skeleton(
    name="subject1_skeleton",
    nodes=["front_left_paw", "body", "front_right_paw"],
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
    subject=subject,
)
skeletons = Skeletons(skeletons=[skeleton])

# create a PoseEstimationSeries for each body part
pose_estimation_series = []
for name in skeleton.nodes[:]:
    pose_estimation_series.append(
        PoseEstimationSeries(
            name=name,
            description=f"Marker placed on {name}.",
            data=np.random.rand(num_frames, 2),  # num_frames x (x, y)
            unit="pixels",
            reference_frame="(0,0) is the top left corner of the video frame.",
            rate=frame_rate,
            confidence=np.random.rand(num_frames),
        )
    )

# the segmentation of this animal needs at most two contours on any frame: an outer boundary, plus
# a hole where the animal curls around a gap. contours have different numbers of vertices from
# frame to frame, so 'data' is padded out to the largest contour and 'vertex_count' records how
# much of each slot is real.
max_contours = 2
max_vertices = 50

contour_data = np.full((num_frames, max_contours, max_vertices, 2), -1, dtype=np.int32)
vertex_count = np.zeros((num_frames, max_contours), dtype=np.uint32)
is_external = np.zeros((num_frames, max_contours), dtype=bool)

rng = np.random.default_rng(0)
for frame in range(num_frames):
    # an outer boundary on every frame, with a vertex count that varies from frame to frame
    n_outer = rng.integers(20, max_vertices + 1)
    contour_data[frame, 0, :n_outer, :] = rng.integers(0, 500, size=(n_outer, 2))
    vertex_count[frame, 0] = n_outer
    is_external[frame, 0] = True

    # a hole on some frames only. the second slot stays padding on the rest, and its
    # vertex_count of 0 is what marks it unused.
    if frame % 3 == 0:
        n_hole = rng.integers(5, 15)
        contour_data[frame, 1, :n_hole, :] = rng.integers(0, 500, size=(n_hole, 2))
        vertex_count[frame, 1] = n_hole
        is_external[frame, 1] = False

contour_series = ContourSeries(
    name="contours",
    description="Outline of the segmented animal, one external boundary plus holes.",
    data=contour_data,
    vertex_count=vertex_count,
    is_external=is_external,
    unit="pixels",
    rate=frame_rate,
)

# store the contours in the same PoseEstimation object as the pose estimates, so both describe
# the same instance of the same subject
pose_estimation = PoseEstimation(
    name="PoseEstimation",
    pose_estimation_series=pose_estimation_series,
    contour_series=[contour_series],
    description="Estimated positions and segmentation contours of subject1.",
    source_software="DeepLabCut",
    source_software_version="2.3.8",
    skeleton=skeleton,
)

behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="processed behavioral data",
)
behavior_pm.add(skeletons)
behavior_pm.add(pose_estimation)

path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the file back and unpad one frame's contours
with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_contours = read_nwbfile.processing["behavior"]["PoseEstimation"].contour_series["contours"]

    frame = 0
    for slot in range(read_contours.data.shape[1]):
        n = read_contours.vertex_count[frame, slot]
        if n == 0:
            continue  # this slot holds no contour on this frame
        vertices = read_contours.data[frame, slot, :n, :]
        boundary = "external" if read_contours.is_external[frame, slot] else "internal"
        print(f"frame {frame}, contour {slot}: {n} vertices, {boundary} boundary")
        print(vertices)
