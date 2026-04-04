# ArogyaDrishti Backend — Sriya Meenakshi Setup

## Step 1 — Install Node.js
Go to nodejs.org → download LTS (v20.x) → install.
Verify: open Command Prompt → `node --version` → should print v20.x.x

## Step 2 — Install dependencies
```bash
cd "E:\New folder\arogyadrishti\backend"
npm install
```

## Step 3 — Set up Firebase
1. Go to firebase.google.com → Go to Console → Add Project → name: `arogyadrishti`
2. Click Build → Firestore Database → Create database → Start in test mode → Region: **asia-south1 (Mumbai)**
3. Click Project Settings (gear icon) → Service Accounts → Generate New Private Key → download JSON
4. Copy values from that JSON into your `.env` file (see Step 5)

**Create these Firestore collections manually (or they auto-create on first write):**
- `patients`
- `screenings`
- `alerts`
- `doctors` — add one document per PHC:
  ```
  { name: "Dr. Ravi Kumar", village: "Pedanandipadu", whatsappNumber: "+919876543210" }
  ```

## Step 4 — Set up Twilio
1. Go to twilio.com → sign up (no credit card needed)
2. Note your **Account SID** and **Auth Token** from the dashboard
3. Click Messaging → WhatsApp → Sandbox → follow instructions
4. Note the sandbox number: **+14155238886**
5. To test: send `join <your-sandbox-keyword>` from your WhatsApp to that number

## Step 5 — Create .env file
```bash
cp .env.example .env
```
Fill in all values in `.env` — never commit this file to GitHub.

## Step 6 — Run locally
```bash
npm run dev
```
Server starts on http://localhost:3000
Test: open http://localhost:3000/api/health → should return `{"status":"ok"}`

## Step 7 — Deploy to Render (free)
1. Push backend folder to GitHub (your branch: `sriya-backend`)
2. Go to render.com → sign up with GitHub
3. Click New + → Web Service → connect your repo
4. Set:
   - Build Command: `npm install`
   - Start Command: `node server.js`
   - Environment: add all variables from your `.env` file
5. Click Create Web Service
6. Copy the live URL (e.g. `https://arogyadrishti-backend.onrender.com`)
7. Give this URL to Koppala Pavani → she updates `AlertApiService.kt`

## Step 8 — Set up ABDM Sandbox
1. Go to sandbox.abdm.gov.in → Register
2. Create an application → note Client ID and Client Secret
3. Add to `.env`: `ABDM_CLIENT_ID` and `ABDM_CLIENT_SECRET`

---

## API Endpoints Summary

| Method | Endpoint | Called by | What it does |
|--------|----------|-----------|-------------|
| POST | `/api/patient/register` | Android app | Creates patient in Firestore |
| GET | `/api/patient/:abha_id` | PHC doctor dashboard | Gets patient + screening history |
| POST | `/api/screening/save` | Android app (WorkManager) | Saves screening results |
| GET | `/api/screening/:id` | PHC doctor dashboard | Gets single screening details |
| POST | `/api/alert/send` | Android app (Red risk) | Sends WhatsApp + SMS via Twilio |
| POST | `/api/abdm/sync` | Android app (online) | Syncs screening to ABDM FHIR R4 |
| GET | `/api/health` | Render health check | Returns 200 OK |

## Alert Flow
```
Android detects Red risk
    → POST /api/alert/send
        → Looks up PHC doctor from Firestore (doctors collection)
        → Sends WhatsApp to doctor (Twilio)
        → Sends SMS to patient in Telugu/Hindi/Tamil
        → Saves alert record in Firestore
        → Cron job checks every hour:
            if phcAcknowledged = false AND sentAt > 48h → sends follow-up WhatsApp
```

## Firebase Free Tier Limits (Spark Plan)
- Storage: 1 GB
- Reads: 50,000/day
- Writes: 20,000/day
- More than enough for prototype/demo phase
