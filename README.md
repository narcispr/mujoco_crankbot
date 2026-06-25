# Crank Bot

Crank Bot is a basic four-legged robot project. The name comes from its black look, but mechanically it is a small quadruped that can be 3D printed and assembled.

The repository includes the STL files in [`assets/`](assets/) and a first basic MuJoCo simulation model.

![Crank Bot](media/crankbot.jpg)

## Bill of Materials

Initial BOM:

- 4x MG995 or MG996 servos
- 4x MG90S servos
- 1x PCA9685 servo driver
- 1x ESP32-C3 microcontroller
- 1x 5 V or 6 V BEC, minimum 5 A, preferably 10 A
- 1x MP1584 buck converter for the microcontroller
- 1x 2S LiPo battery
- 3D printed parts from [`assets/`](assets/)

## Simulation

The repository currently contains a basic MuJoCo XML model of the robot. A Python script will be added later to interact with the simulation.

![Crank Bot in MuJoCo](media/crankbot_mujoco.png)

## Control del robot real amb ESP32-C3

Aquesta versió substitueix el control Arduino bàsic de [`firmware/robo_dog_sketch.ino`](firmware/robo_dog_sketch.ino) per un bridge WiFi amb ESP32-C3:

- Firmware ESP32-C3: [`firmware/esp32_c3_servo_bridge/esp32_c3_servo_bridge.ino`](firmware/esp32_c3_servo_bridge/esp32_c3_servo_bridge.ino)
- Client PC: [`scripts/robot_client.py`](scripts/robot_client.py)
- Presets visibles: [`scripts/config/robot_presets.json`](scripts/config/robot_presets.json)

El PC envia 8 setpoints normalitzats en l'interval `[-1, 1]` per UDP. L'ESP els converteix a polsos PWM per al PCA9685 i limita la velocitat de canvi amb els paràmetres `minRpm` i `maxRpm` definits per cada servo. Això és un limitador de consigna: els MG995/MG996/MG90S no donen feedback de posició, per tant l'ESP no pot mesurar la RPM real.

Ordre dels 8 motors:

| Índex | Motor | Canal PCA9685 inicial |
| ---: | --- | ---: |
| 0 | `front_left_shoulder` | 8 |
| 1 | `front_left_elbow` | 9 |
| 2 | `front_right_shoulder` | 12 |
| 3 | `front_right_elbow` | 13 |
| 4 | `back_left_shoulder` | 10 |
| 5 | `back_left_elbow` | 11 |
| 6 | `back_right_shoulder` | 14 |
| 7 | `back_right_elbow` | 15 |

### Connexions

Connexions recomanades:

- PCA9685 `V+` a la font/BEC de servos de 5-6 V.
- PCA9685 `GND`, ESP32-C3 `GND` i GND de la font de servos units.
- PCA9685 `VCC` a `3V3` de l'ESP32-C3.
- PCA9685 `SDA` al GPIO `8` de l'ESP32-C3.
- PCA9685 `SCL` al GPIO `9` de l'ESP32-C3.
- Servos als canals `8..15` del PCA9685 segons la taula anterior.

No alimentis els servos des de l'USB ni des del pin `5V` de l'ESP. Els servos han de tenir una alimentació separada amb prou corrent, i només han de compartir GND amb l'ESP.

Si la teva placa ESP32-C3 no exposa I2C als GPIO `8/9`, canvia aquests valors al firmware:

```cpp
const int I2C_SDA_PIN = 8;
const int I2C_SCL_PIN = 9;
```

### Instal·lació de l'Arduino IDE

1. Instal·la Arduino IDE 2.x.
2. A `File > Preferences`, afegeix aquest URL a `Additional boards manager URLs`:

   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```

3. A `Tools > Board > Boards Manager`, instal·la `esp32` by Espressif Systems.
4. A `Library Manager`, instal·la:
   - `Adafruit PWM Servo Driver Library`
   - `Adafruit BusIO`
5. Obre [`firmware/esp32_c3_servo_bridge/esp32_c3_servo_bridge.ino`](firmware/esp32_c3_servo_bridge/esp32_c3_servo_bridge.ino).
6. A `Tools > Board`, tria la teva placa ESP32-C3. Si no surt exactament, prova `ESP32C3 Dev Module`.
7. Connecta l'ESP per USB i tria el port a `Tools > Port`.
8. Prem `Upload`.

Si la placa no entra en mode programació, mantén premut `BOOT`, prem i deixa anar `RESET`, i torna a pujar el firmware.

### Configuració WiFi de l'ESP

Per defecte, l'ESP crea una xarxa pròpia:

- SSID: `crankbot-esp32`
- Password: `crankbot123`
- IP de l'ESP: `192.168.4.1`
- Port UDP: `4210`

Connecta el PC a aquesta WiFi abans d'executar el client Python.

Si prefereixes que l'ESP entri a la teva WiFi, omple aquests camps al firmware:

```cpp
const char *WIFI_SSID = "nom_de_la_wifi";
const char *WIFI_PASSWORD = "password";
```

En aquest mode, obre el Serial Monitor a `115200` baud per veure la IP assignada i passa-la al client amb `--host`.

### Calibratge dels servos

Al firmware, cada servo té:

```cpp
{"front_left_shoulder", 8, 500, 1500, 2750, 2.0, 50.0, false}
```

Els camps són:

| Camp | Significat |
| --- | --- |
| Nom | Nom del joint |
| Canal | Canal del PCA9685 |
| `minUs` | Pols per a setpoint `-1` |
| `centerUs` | Pols per a setpoint `0` |
| `maxUs` | Pols per a setpoint `1` |
| `minRpm` | Pas mínim quan hi ha moviment |
| `maxRpm` | Velocitat màxima de canvi de consigna |
| `invert` | Inverteix el signe del setpoint |

Els valors inicials surten del sketch Arduino antic: `500 us`, `2500 us`, `2750 us` i centre `1500 us`. Abans d'executar la policy, prova motor a motor amb valors petits, per exemple `-0.2`, `0.0`, `0.2`, i ajusta `minUs`, `centerUs`, `maxUs` i `invert`.

### Client Python

Des del PC principal, crea l'entorn Python i instal·la dependències:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per veure els presets disponibles:

```bash
python scripts/robot_client.py list-presets
```

Per enviar una posició predefinida:

```bash
python scripts/robot_client.py preset center
python scripts/robot_client.py preset stand_approx --duration 3
python scripts/robot_client.py preset folded_approx --duration 3
```

Per defecte, aquests comandaments també obren MuJoCo i mostren el mateix moviment al simulador. Si només vols enviar a l'ESP sense simulador:

```bash
python scripts/robot_client.py --disable-mujoco preset stand_approx --duration 3
```

Per enviar 8 valors manuals:

```bash
python scripts/robot_client.py manual 0 0 0 0 0 0 0 0
python scripts/robot_client.py manual 0.2 0 -0.2 0 0.2 0 -0.2 0 --duration 2
```

En mode manual amb MuJoCo actiu, els 8 valors manuals són el target del simulador. El client llegeix contínuament les posicions reals dels joints simulats (`qpos`) i envia aquestes posicions a l'ESP. Per saltar aquest comportament i enviar directament els 8 valors manuals:

```bash
python scripts/robot_client.py --disable-mujoco manual 0.2 0 -0.2 0 0.2 0 -0.2 0 --duration 2
```

Per comprovar comunicació amb l'ESP:

```bash
python scripts/robot_client.py command PING
python scripts/robot_client.py command STATUS
python scripts/robot_client.py command STOP
```

Si l'ESP està connectat a una WiFi externa, usa la IP que surt pel Serial Monitor:

```bash
python scripts/robot_client.py --host 192.168.1.123 command STATUS
```

### Executar una policy al robot

El client pot carregar una policy SAC de Stable-Baselines3 i enviar les accions al robot:

```bash
python scripts/robot_client.py policy --model logs/crankbot_walk_gym/run_xxx/final_model.zip
```

Si no passes `--model`, busca l'últim model sota `logs/crankbot_walk_gym`.

El programa demana:

- `Goal range in meters`: distància inicial al goal.
- `Goal bearing in degrees`: angle relatiu del goal respecte al robot.
- `Maximum execution time in seconds`: temps màxim d'execució.

També pots passar-ho sense prompts:

```bash
python scripts/robot_client.py policy \
  --model logs/crankbot_walk_gym/run_xxx/final_model.zip \
  --goal-range 0.5 \
  --goal-bearing-deg 0 \
  --max-time 8
```

Abans d'executar la policy, el client envia `q_stand` durant `--prime-duration`, perquè les policies s'han entrenat sortint d'una postura propera a `q_stand`.

Amb MuJoCo actiu, que és el comportament per defecte, el range i bearing inicials creen un goal visible al `goal_site` del simulador. A partir d'aquí, cada crida a la policy usa el range i bearing actuals entre el robot simulat i aquest goal. El feedback del goal ve del simulador, no del robot real.

Si vols executar la policy sense obrir ni avançar MuJoCo:

```bash
python scripts/robot_client.py --disable-mujoco policy \
  --model logs/crankbot_walk_gym/run_xxx/final_model.zip \
  --goal-range 0.5 \
  --goal-bearing-deg 0 \
  --max-time 8
```

Per al robot real, entrena o carrega una policy amb observacions d'actor, és a dir amb `--disable-privileged`. Una policy entrenada amb l'observació privilegiada de 80 valors espera estat de simulador que el robot real no té.

Important: el control dels servos reals continua sent open-loop. El client sap les consignes que ha enviat, l'historial d'accions, la fase de la marxa i, si MuJoCo està actiu, el goal respecte al robot simulat. No sap si el robot real ha avançat o ha girat. Si afegeixes odometria, càmera o localització externa, s'ha d'actualitzar `goal_range` i `goal_bearing` dins de [`scripts/robot_client.py`](scripts/robot_client.py).


## Training

The Gymnasium environment is [`scripts/crankbot_walk_gym_env.py`](scripts/crankbot_walk_gym_env.py). It wraps the CPU MuJoCo vectorized environment as a single-env Gym API for SAC training.

| Mode | Observation size | Contents |
| --- | ---: | --- |
| Default | 80 | Actor observation + privileged simulator state |
| `--disable-privileged` | 68 | Actor observation only |

Actor observation:

| Term | Size | Meaning |
| --- | ---: | --- |
| `q_cmd_history` | 32 | 4-step history of 8 commanded joint targets, as `q_cmd - q_stand` |
| `action_history` | 32 | 4-step history of 8 normalized actions |
| `goal_bearing / pi` | 1 | Relative goal direction |
| `goal_range / target_goal_range` | 1 | Normalized distance to the goal |
| `sin(phase), cos(phase)` | 2 | Gait phase signal |

Standing reference pose:

| Joint order | `front_left_shoulder` | `front_left_elbow` | `front_right_shoulder` | `front_right_elbow` | `back_left_shoulder` | `back_left_elbow` | `back_right_shoulder` | `back_right_elbow` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_stand` rad | -1.15 | -2.34 | 1.15 | 2.34 | 2.00 | -2.34 | -2.00 | 2.34 |

Privileged extra observation:

| Term | Size | Meaning |
| --- | ---: | --- |
| `base_velocity_body` | 3 | Local base velocity, including yaw rate |
| `base_z` | 1 | Base height |
| `roll_pitch` | 2 | Base tilt |
| `foot_contacts` | 4 | Foot contact flags |
| `leg_contact` | 1 | Upper-leg ground contact |
| `body_contact` | 1 | Base ground contact |

Actions are 8 normalized joint command increments:

$$
q_{cmd,t+1} = clip(q_{cmd,t} + a_t \cdot action\_scale,\ q_{min},\ q_{max}),\quad a_t \in [-1, 1]^8
$$

Reward:

$$
r = -tanh(goal\_range / goal\_reward\_scale) - c_s ||a_t-a_{t-1}||^2 - c_a ||a_t||^2 - p_{idle} - p_{leg} - p_{lower} - p_{body} - p_{fall}
$$

| Term | Meaning |
| --- | --- |
| $-tanh(goal\_range / goal\_reward\_scale)$ | Goal progress term; less negative as the robot gets closer to the target |
| $-c_s \|\|a_t-a_{t-1}\|\|^2$ | Smoothness penalty for abrupt action changes |
| $-c_a \|\|a_t\|\|^2$ | Action magnitude penalty |
| $-p_{idle}$ | Penalty for sending nearly zero action while the goal is not reached |
| $-p_{leg}$ | Penalty when upper-leg collision geoms touch the floor |
| $-p_{lower}$ | Penalty when lower-leg collision geoms touch the floor |
| $-p_{body}$ | Penalty when the base touches the floor |
| $-p_{fall}$ | Fall penalty when base height is below the fall threshold |

After training with SAC, this behavior has been obtained:



[![Watch the video](media/youtube.png)](https://youtu.be/LTj4IBt9pyA)



## Future Work

- Include some dynamics/noise randomization in the learning.
- Execute the trained policy in the real hardware.
