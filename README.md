# Tropical Cyclone Tracking

These examples show how to run the TC tracker, originally ported from **TempestExtremes**, by integrating **TECA** into a typical **Earth2Studio** workflow.

## ARCO data source

```python
import torch
import numpy as np
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

## SFNO model and 2 initial conditions

```python
import torch
import numpy as np
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.px import SFNO
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords, CoordSystem
from tqdm import tqdm

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

start_time_1 = datetime(2023, 8, 5)  # Start date for inference
start_time_2 = datetime(2023, 1, 1)  # Start date for inference
nsteps = 10  # Number of steps to run the tracker for into future
times_1 = [start_time_1 + timedelta(hours=6 * i) for i in range(nsteps+1)]
times_2 = [start_time_2 + timedelta(hours=6 * i) for i in range(nsteps+1)]
da = data([start_time_1, start_time_2], ['z'])
x_data, coords_data = prep_data_array(da, device=device)

# Load the initial state
x, coords = fetch_data(
    source=data,
    time=to_time_array([start_time_1, start_time_2]),
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
             "time": np.array([np.datetime64(times_1[step], 'ns'), np.datetime64(times_2[step], 'ns')]),
             "variable": variables,
             "lat": coords_data["lat"],
             "lon": coords_data["lon"],
        })
        # Run tracker
        x, coords = map_coords(x, coords, tracker.detect.input_coords())
        tracker.detect._current_time = np.array([np.datetime64(times_1[step], 'ns'), np.datetime64(times_2[step], 'ns')])
        output, output_coords = tracker.detect(x, coords)
        pbar.update(1)
        if step == nsteps:
            break

tracker.stitch._nsteps = nsteps+1
out, out_coords = tracker.stitch(output, output_coords)
```

## SFNO Ensemble

```python
import torch
import numpy as np
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.px import SFNO
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble
from earth2studio.data import prep_data_array
from earth2studio.utils.coords import CoordSystem

# Create the data source
data = ARCO()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
model = SFNO.load_model(package)
model = model.to(device)

# Instantiate the pertubation method
sg = SphericalGaussian(noise_amplitude=0.15)

io = ZarrBackend()

nsteps = 10
nensemble = 4
io_model = ensemble(["2009-08-05"], nsteps, nensemble, model, data, io, sg,
                    batch_size=2,
                    output_coords={"variable": np.array(["msl", "u10m", "v10m", "z300", "z500"])},
                    device=device)

# the model data is on CPU after running
msl  = io_model["msl"][:]
u10m = io_model["u10m"][:]
v10m = io_model["v10m"][:]
z300 = io_model["z300"][:]
z500 = io_model["z500"][:]

# shape: (nensemble, time, nsteps+1, lat, lon)
# shape: [4, 1, 11, 721, 1440]
msl_tensor  = torch.tensor(msl,  dtype=torch.float32)
u10m_tensor = torch.tensor(u10m, dtype=torch.float32)
v10m_tensor = torch.tensor(v10m, dtype=torch.float32)
z300_tensor = torch.tensor(z300, dtype=torch.float32)
z500_tensor = torch.tensor(z500, dtype=torch.float32)

device = torch.device("cpu")

start_time = datetime(2009, 8, 5)  # Start date for inference
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]
da = data(start_time, ['z'])
x_data, coords_data = prep_data_array(da, device=device) # shape: [1, 1, 721, 1440]
x_data = x_data.unsqueeze(2)  # shape: [1, 1, 1, 721, 1440]
z_tensor = x_data.expand(nensemble, 1, nsteps+1, 721, 1440)  # shape: [nensemble, time, nsteps+1, lat, lon]

# Stack along new variable dimension (dim=3)
# shape: (nensemble, time, nsteps+1, nvariables, lat, lon)
# shape: [4, 1, 11, 6, 721, 1440]
x_combined = torch.stack([msl_tensor, u10m_tensor, v10m_tensor, z_tensor, z300_tensor, z500_tensor], dim=3)

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
tracker = tracker.to(device)
tracker.detect._device = device

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
```

## FCN3 Ensemble

```python
import torch
import numpy as np
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.px import FCN3
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble as run
from earth2studio.data import prep_data_array
from earth2studio.utils.coords import CoordSystem

# Create the data source
data = ARCO()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the default model package
model = FCN3.load_model(FCN3.load_default_package())
model = model.to(device)

# no perturbation required due to hidden Markov formulation of FCN3
perturbation = Zero()

io = ZarrBackend()

nsteps = 10
nensemble = 4
# invoke inference with 4 ensemble members
run(time=["2009-08-05"], nsteps=nsteps, nensemble=nensemble, prognostic=model, data=data, io=io, perturbation=perturbation,
    batch_size=1, output_coords={"variable": np.array(["msl", "u10m", "v10m", "z300", "z500"])}, device=device,)

# the model data is on CPU after running
msl  = io["msl"][:]
u10m = io["u10m"][:]
v10m = io["v10m"][:]
z300 = io["z300"][:]
z500 = io["z500"][:]

# shape: (nensemble, time, nsteps+1, lat, lon)
# shape: [4, 1, 11, 721, 1440]
msl_tensor  = torch.tensor(msl,  dtype=torch.float32)
u10m_tensor = torch.tensor(u10m, dtype=torch.float32)
v10m_tensor = torch.tensor(v10m, dtype=torch.float32)
z300_tensor = torch.tensor(z300, dtype=torch.float32)
z500_tensor = torch.tensor(z500, dtype=torch.float32)

device = torch.device("cpu")

start_time = datetime(2009, 8, 5)  # Start date for inference
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]
da = data(start_time, ['z'])
x_data, coords_data = prep_data_array(da, device=device) # shape: [1, 1, 721, 1440]
x_data = x_data.unsqueeze(2)  # shape: [1, 1, 1, 721, 1440]
z_tensor = x_data.expand(nensemble, 1, 11, 721, 1440)  # shape: [nensemble, time, nsteps+1, lat, lon]

# Stack along new variable dimension (dim=3)
# shape: (nensemble, time, nsteps+1, nvariables, lat, lon)
# shape: [4, 1, 11, 6, 721, 1440]
x_combined = torch.stack([msl_tensor, u10m_tensor, v10m_tensor, z_tensor, z300_tensor, z500_tensor], dim=3)

# Create tropical cyclone tracker
tracker = teca_tempest_tc_detect()
tracker = tracker.to(device)
tracker.detect._device = device

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
