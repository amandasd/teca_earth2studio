# Tropical Cyclone Tracking

These examples show how to run the TC tracker, originally ported from **TempestExtremes**, by integrating **TECA** into a typical **Earth2Studio** workflow.
This version is developed for only 1 batch.

## From ARCO data source

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
    tracker.detect._current_time = np.array([np.datetime64(times[step], 'ns')])
    output, output_coords = tracker.detect(x, coords)
    
tracker.stitch._nsteps = nsteps
out, out_coords = tracker.stitch(output, output_coords)
```

## From SFNO model

```python
import torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.px import SFNO
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords
from tqdm import tqdm
import numpy as np

# Create the data source
data = ARCO()

start_time = datetime(2009, 8, 5)  # Start date for inference
nsteps = 10  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps)]

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tracker = tracker.to(device)
tracker.detect._device = device

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
prognostic = SFNO.load_model(package)
prognostic = prognostic.to(device)

# Load the initial state
x, coords = fetch_data(
    source=data,
    time=to_time_array([start_time]),
    variable=prognostic.input_coords()["variable"],
    lead_time=prognostic.input_coords()["lead_time"],
    device=device,
)

# Create prognostic iterator
model = prognostic.create_iterator(x, coords)

with tqdm(total=nsteps, desc="Running inference") as pbar:
    for step, (x, coords) in enumerate(model):
        # Run tracker
        x, coords = map_coords(x, coords, tracker.detect.input_coords())
        tracker.detect._current_time = np.array([np.datetime64(times[step], 'ns')])
        output, output_coords = tracker.detect(x, coords)
        # lets remove the lead time dim
        output = output[:, 0]; del output_coords["lead_time"]
        pbar.update(1)
        if step == nsteps-1:
            break

tracker.stitch._nsteps = nsteps
out, out_coords = tracker.stitch(output, output_coords)


