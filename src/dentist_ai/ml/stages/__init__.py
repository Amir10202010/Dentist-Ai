"""Analysis stages.

Each module here implements :class:`dentist_ai.ml.pipeline.Stage` for one part
of reading a CBCT. They are assembled into a pipeline in
:mod:`dentist_ai.ml.cbct`, which is the only module that knows the order.

The shared frame of reference every stage assumes is the DICOM patient
coordinate system with a standard head orientation:

* ``z`` — slice index. Low is inferior, high is superior, so the mandible
  occupies the lower slices and the maxilla the upper ones.
* ``y`` — row within a slice. Low is anterior, high is posterior.
* ``x`` — column within a slice. Low is the patient's right, high their left.

That holds for a correctly positioned acquisition and is what the ingest
sorter produces. A rotated or mis-oriented scan degrades the *regional*
labelling — a finding lands in the wrong quadrant — without affecting the
detection itself, which is geometry-free. The regional label is presented as
editable for exactly that reason.
"""
