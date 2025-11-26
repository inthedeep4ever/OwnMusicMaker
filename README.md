## 1. Benodigdheden (downloads & installs)

## Vereist om dit project te draaien

### 1. **Docker Desktop**

Download & installeer:

* Windows: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
* Mac: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
* Linux: jouw package manager

Zorg dat Docker Desktop draait vóór je verder gaat.

### 2. **Git** (om de repository te clonen)

Download:

* [https://git-scm.com/downloads](https://git-scm.com/downloads)

Check of het werkt:

```
git --version
```

### 3. **Python (optioneel)**

Alle Python dependencies worden in Docker geïnstalleerd, maar handig voor testen.

---

## 2. Repository downloaden (clone)

Open een terminal en run:

```
git clone https://github.com/<jouw-repo>/techno-ai.git
cd techno-ai
```

(Als je nog geen repo hebt, kun je de projectmap direct kopiëren.)

---

## 3. Projectstructuur (overzicht)

```
techno-ai/
│
├── docker-compose.yml
├── nginx/
│   └── nginx.conf
│
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py
│   ├── model.py
│   ├── dataset.py
│   ├── generate.py
│   │
│   ├── uploads/        # WAV's die jij uploadt
│   ├── generated/      # Audio die AI genereert
│   └── checkpoints/    # Model weights
│
└── README.md
```

Belangrijk:

* **uploads/** → hier dump je je eigen Ableton WAV’s
* **generated/** → hier komt de gegenereerde muziek terecht

---

## 4. Vereiste Docker & Python tools (requirements)

Deze worden automatisch geïnstalleerd in de container, maar staan hier ter referentie.

### `app/requirements.txt`

```
fastapi
uvicorn
pydantic
torchaudio
torch
numpy
librosa
python-multipart
```

Dockerfile installeert dit automatisch via pip.

---

## 5. Project starten (Docker build + run)

Open een terminal in de projectmap en voer uit:

```
docker-compose up --build
```

Dit doet het volgende:

* bouwt de AI-container (PyTorch + FastAPI)
* start de API op *localhost:8000*
* start Nginx reverse proxy op *localhost:80*
* koppelt volumes:

  * `uploads → /app/uploads`
  * `generated → /app/generated`
  * `checkpoints → /app/checkpoints`

Je ziet logs zoals:

```
api     | Uvicorn running on 0.0.0.0:8000
nginx   | nginx started
```

Nu draait alles.

---

## 6. WAV-bestanden uploaden (jouw Ableton audio)

Je kunt je WAV’s op twee manieren uploaden:

## Methode A — Slepen in de map

Plaats bestanden in:

```
techno-ai/app/uploads/
```

## Methode B — Via API request

Gebruik Postman of een webclient:

```
POST http://localhost/api/upload
```

Form-data:

```
key: file
value: jouwfile.wav
```

Nginx maakt deze beschikbaar op:

```
http://localhost/uploads/
```

---

## 7. Nieuwe techno audio genereren

Trigger de AI met:

```
GET http://localhost/api/generate
```

Dit doet het volgende:

* Laadt `model.py`
* Maakt random latent techno textures
* Bouwt nieuw audiofragment via `generate.py`
* Slaat het op als:

```
app/generated/output.wav
```

Je krijgt terug:

```json
{
  "status": "done",
  "file": "generated/output.wav"
}
```

Beluister meteen via browser:

```
http://localhost/generated/output.wav
```

---

## 8. Hoe de AI werkt (kort uitgelegd)

* Jouw WAV’s worden ingelezen via **TechnoDataset**
* Audio wordt gefilterd & genormaliseerd
* Model is een **1D Convolutional Autoencoder** → goed voor techno textures
* AI leert:

  * kicks
  * rumbles
  * basslines
  * noise textures
  * percussive hits
  * stab patterns
* De decoder genereert nieuwe audio based on learned latent space
* Output is 44.1kHz en direct bruikbaar in Ableton

---

## 9. Waar staat alles opgeslagen?

### ✔ Jouw uploads

```
app/uploads/
```

### AI generaties

```
app/generated/
```

### Model-weights

```
app/checkpoints/
```

Door Docker volumes blijven deze bewaard **zelfs als containers stoppen**.

---

## 10. Project stoppen

In de terminal waar Docker draait druk je:

```
CTRL + C
```

Of:

```
docker-compose down
```

---

## 11. Extra (optioneel)

Wil je dat ik toevoeg:

* volledig **trainingsscript** met epochs & retraining
* **web dashboard** (upload + generate + player)
* **meer geavanceerd model** (Diffusion / GAN voor hogere kwaliteit)
* **Ableton Max for Live integratie**
* **database + user system**

Zeg het — ik breid het project verder uit.

---

## Klaar voor gebruik

Met deze README kun je:

* WAV’s uploaden
* AI techno genereren
* Output meteen in Ableton slepen
* Je eigen AI-model bouwen op jouw eigen sound

Veel plezier met produceren! 
