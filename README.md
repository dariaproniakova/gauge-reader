# gauge-reader

Reading analog pressure/temperature gauges from camera images for autonomous robotic
inspection: detect the gauge, find the needle, convert its angle to a value.

> 🚧 **Work in progress.** Pipeline and results are being added; see the roadmap below.

<!-- Replace with a demo GIF or an example image: input → detected gauge → read value -->
_Demo image coming soon._

## Results

| Method | Mean abs. error (% of full scale) | Reading success rate | FPS |
|---|---|---|---|
| Classical baseline (Hough circles + line) | – | – | – |
| YOLO detection + needle keypoints | – | – | – |

## Approach

1. **Detect** the gauge face in the image (YOLO).
2. **Rectify** the dial (ellipse fit → perspective correction) so it appears circular.
3. **Locate** the needle and the min/max scale marks (keypoints or line detection).
4. **Convert** the needle angle to a reading by interpolating between the scale ends.
5. **Deploy** as a ROS 2 node that subscribes to a camera topic and publishes readings.

## How to run

```bash
git clone https://github.com/dariaproniakova/gauge-reader.git
cd gauge-reader
pip install -r requirements.txt
pip install -e .
python -m gauge_reader.read --image path/to/gauge.jpg --min 0 --max 10
```

## Project structure

```
src/gauge_reader/   detection, geometry and reading logic
notebooks/          exploration and evaluation
results/            final figures and metrics
ros2/               ROS 2 node (planned)
tests/              unit tests
data/README.md      how to get the data
```

## Roadmap

- [x] Repository structure, angle → value conversion with tests
- [ ] Dataset collection and labelling
- [ ] Classical baseline
- [ ] YOLO gauge detector + needle keypoints
- [ ] Evaluation on held-out images
- [ ] ROS 2 node + Docker image

## What I learned / limitations

*To be written once the evaluation is finished.*

## License

MIT
