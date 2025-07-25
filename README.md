# Tropical Cyclone Tracking

This example shows how to run the TC tracker, originally ported from **TempestExtremes**, by integrating **TECA** into a typical **Earth2Studio** workflow.
This version is developed for only 1 batch.

```python
import torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.data import prep_data_array

# Create the data source
data = ARCO()

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tracker = tracker.to(device)
tracker.detect._device = device

start_time = datetime(2009, 8, 5)  # Start date
nsteps = 10  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps)]

for step, time in enumerate(times):
    da = data(time, tracker.detect.input_coords()["variable"])
    x, coords = prep_data_array(da, device=device)
    current_time = coords.get("time")
    tracker.detect._current_time = current_time    
    output, output_coords = tracker.detect(x, coords)
    
tracker.stitch._nsteps = nsteps
out, out_coords = tracker.stitch(output, output_coords)
