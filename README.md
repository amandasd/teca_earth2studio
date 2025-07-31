# Tropical Cyclone Tracking

These examples show how to run the TC tracker, originally ported from **TempestExtremes**, by integrating **TECA** into a typical **Earth2Studio** workflow.

## From ARCO data source

```python
import torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.data import prep_data_array

# Create the data source
data = ARCO()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
tracker = tracker.to(device)
tracker.detect._device = device
tracker.detect.reset_path_buffer()
tracker.detect.reset_step()

start_time = datetime(2009, 8, 5)  # Start date
nsteps = 10  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]

for step, time in enumerate(times):
    da = data(time, tracker.detect.input_coords()["variable"])
    x, coords = prep_data_array(da, device=device)
    tracker.detect._current_time = np.array([np.datetime64(time, 'ns')])
    output, output_coords = tracker.detect(x, coords)
    
tracker.stitch._nsteps = nsteps+1
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
from earth2studio.utils.coords import map_coords, CoordSystem
from tqdm import tqdm
import numpy as np

# Create the data source
data = ARCO()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
tracker = tracker.to(device)
tracker.detect._device = device
tracker.detect.reset_path_buffer()
tracker.detect.reset_step()

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
prognostic = SFNO.load_model(package)
prognostic = prognostic.to(device)

start_time = datetime(2009, 8, 5)  # Start date for inference
nsteps = 10  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]
da = data(start_time, ['z'])
x_data, coords_data = prep_data_array(da, device=device)

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
    for step, (x_sfno, coords_sfno) in enumerate(model):
        # lets remove the lead time dim
        x_sfno_squeezed = x_sfno.squeeze(1); del coords_sfno["lead_time"]
        x = torch.cat((x_data, x_sfno_squeezed), dim=1)
        variables = np.concatenate([coords_data["variable"], coords_sfno["variable"]])
        coords = CoordSystem({
             "time": np.array([np.datetime64(times[step], 'ns')]),
             "variable": variables,
             "lat": coords_data["lat"],
             "lon": coords_data["lon"],
        })
        # Run tracker
        x, coords = map_coords(x, coords, tracker.detect.input_coords())
        tracker.detect._current_time = np.array([np.datetime64(times[step], 'ns')])
        output, output_coords = tracker.detect(x, coords)
        pbar.update(1)
        if step == nsteps:
            break

tracker.stitch._nsteps = nsteps+1
out, out_coords = tracker.stitch(output, output_coords)
```

## From SFNO Ensemble

```python
import torch
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.px import SFNO
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble
from earth2studio.data import prep_data_array
from earth2studio.utils.coords import CoordSystem
import numpy as np

# Create the data source
data = ARCO()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
tracker = tracker.to(device)
tracker.detect._device = device

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
model = SFNO.load_model(package)
model = model.to(device)

start_time = datetime(2009, 8, 5)  # Start date for inference
nsteps = 10  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]

# Instantiate the pertubation method
sg = SphericalGaussian(noise_amplitude=0.15)

io = ZarrBackend()

nensemble = 4
io_model = ensemble(["2009-08-05"], nsteps, nensemble, model, data, io, sg,
                    batch_size=2,
                    output_coords={"variable": np.array(["msl", "u10m", "v10m", "z300", "z500"])},)

msl  = io_model["msl"][:]
u10m = io_model["u10m"][:]
v10m = io_model["v10m"][:]
z300 = io_model["z300"][:]
z500 = io_model["z500"][:]

# shape: (ensemble, time, nsteps, lat, lon)
msl_tensor  = torch.tensor(msl, dtype=torch.float32, device=device)
u10m_tensor = torch.tensor(u10m, dtype=torch.float32, device=device)
v10m_tensor = torch.tensor(v10m, dtype=torch.float32, device=device)
z300_tensor = torch.tensor(z300, dtype=torch.float32, device=device)
z500_tensor = torch.tensor(z500, dtype=torch.float32, device=device)

da = data(start_time, ['z'])
x_data, coords_data = prep_data_array(da, device=device) # shape: [1, 1, 721, 1440]
x_data = x_data.unsqueeze(2)  # shape: [1, 1, 1, 721, 1440]
z_tensor = x_data.expand(4, 1, 11, 721, 1440)  # shape: [4, 1, 11, 721, 1440]

# Stack along new variable dimension (dim=3)
x_combined = torch.stack([msl_tensor, u10m_tensor, v10m_tensor, z_tensor, z300_tensor, z500_tensor], dim=3)

member_outputs = []
member_outputs_coords = []
tracker.stitch._nsteps = nsteps+1
for ens in range(x_combined.shape[0]):  # loop over ensemble dimension
    tracker.detect.reset_path_buffer()
    tracker.detect.reset_step()
    for step, time in enumerate(times):
        coords = CoordSystem({
                    "time": np.array([np.datetime64(time, 'ns')]),
                    "variable": np.array(["msl", "u10m", "v10m", "z", "z300", "z500"]),
                    "lat": np.linspace(90, -90, 721, endpoint=True),
                    "lon": np.linspace(0, 360, 1440, endpoint=False),
                })
        tracker.detect._current_time = np.array([np.datetime64(time, 'ns')])
        output, output_coords = tracker.detect(x_combined[ens,:,step,:,:,:], coords)
    out, out_coords = tracker.stitch(output, output_coords)
    member_outputs.append(out)
    member_outputs_coords.append(out_coords)
