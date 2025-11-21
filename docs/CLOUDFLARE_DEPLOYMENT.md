# TryCloudflare Setup Guide

This guide explains how to run the xquizit interview system with **TryCloudflare** for instant external HTTPS access.

## What is TryCloudflare?

TryCloudflare is Cloudflare's **free, instant tunnel service** that:
- ✅ Requires **NO account creation**
- ✅ Requires **NO authentication**
- ✅ Requires **NO configuration files**
- ✅ Requires **NO DNS setup**
- ✅ Generates **random HTTPS URLs** automatically
- ✅ Works globally from any device
- ✅ Enables microphone/camera access (secure context)

**Perfect for:** Quick demos, testing, development, showing work to others

---

## Quick Start

### Option 1: Start Everything at Once (Recommended)

Double-click `start-all-trycloudflare.bat` or run:
```bash
start-all-trycloudflare.bat
```

This opens 4 windows:
1. **Backend Server** - FastAPI running on localhost:8000
2. **Frontend Dev Server** - Vite running on localhost:5173
3. **Backend Tunnel** - Shows public HTTPS URL for the API
4. **Frontend Tunnel** - Shows public HTTPS URL for the frontend

### Option 2: Start Services Individually

```bash
# Start backend
start-backend.bat

# Start frontend
start-frontend.bat

# Start backend tunnel
start-backend-tunnel.bat

# Start frontend tunnel
start-frontend-tunnel.bat
```

---

## Complete Setup Workflow

### Step 1: Start All Services

Run `start-all-trycloudflare.bat` and wait for all 4 windows to open.

### Step 2: Get the Backend Tunnel URL

Look at the **"Backend Tunnel"** window. You'll see output like:

```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://seasonal-deck-organisms-sf.trycloudflare.com                                       |
+--------------------------------------------------------------------------------------------+
```

**Copy this URL!** (e.g., `https://seasonal-deck-organisms-sf.trycloudflare.com`)

### Step 3: Update Frontend Configuration

Open `frontend/src/config.js` and find these lines:

```javascript
// Local development (default)
export const API_BASE_URL = 'http://localhost:8000';
export const WS_BASE_URL = 'ws://localhost:8000';
```

**Comment them out** and add your Backend Tunnel URL:

```javascript
// Local development (default)
// export const API_BASE_URL = 'http://localhost:8000';
// export const WS_BASE_URL = 'ws://localhost:8000';

// TryCloudflare (for external access)
export const API_BASE_URL = 'https://seasonal-deck-organisms-sf.trycloudflare.com';
export const WS_BASE_URL = 'wss://seasonal-deck-organisms-sf.trycloudflare.com';
```

**Important:**
- Use `https://` for API_BASE_URL
- Use `wss://` for WS_BASE_URL (secure WebSocket)

### Step 4: Restart Frontend

Close the **"Frontend Dev Server"** window and restart it:
```bash
start-frontend.bat
```

Or just press `Ctrl+C` in that window and run `npm run dev` again.

### Step 5: Get the Frontend Tunnel URL

Look at the **"Frontend Tunnel"** window. You'll see a URL like:

```
https://random-name-xyz.trycloudflare.com
```

### Step 6: Access Your Application

Open a browser (from **any device**, anywhere in the world) and visit the **Frontend Tunnel URL**.

You should see:
- ✅ Your interview application loads
- ✅ Green lock icon (valid HTTPS)
- ✅ No certificate warnings
- ✅ **Microphone access works!**

---

## Important Notes

### URLs Change on Every Restart

⚠️ **TryCloudflare generates random URLs each time you start the tunnels.**

This means:
- Every time you restart the tunnels, you get **new random URLs**
- You must **update `frontend/src/config.js`** with the new Backend URL
- You must **restart the frontend** to pick up the new configuration
- You must **share the new Frontend URL** with anyone who needs to access the app

**Example workflow:**
```
Day 1: Backend URL = https://abc-123.trycloudflare.com
       Frontend URL = https://xyz-789.trycloudflare.com

[Restart tunnels]

Day 2: Backend URL = https://def-456.trycloudflare.com  ← Different!
       Frontend URL = https://uvw-012.trycloudflare.com  ← Different!
       → Must update config.js and restart frontend
```

### Local Development

If you're only testing locally (not sharing with others), you don't need tunnels:

1. **Comment out** the TryCloudflare URLs in `frontend/src/config.js`
2. **Uncomment** the localhost URLs:
   ```javascript
   export const API_BASE_URL = 'http://localhost:8000';
   export const WS_BASE_URL = 'ws://localhost:8000';
   ```
3. Start only backend and frontend:
   ```bash
   start-backend.bat
   start-frontend.bat
   ```
4. Access at `http://localhost:5173`

**Note:** Microphone only works on `localhost`, not on network IPs without HTTPS.

---

## Stopping the Application

Double-click `stop-all-trycloudflare.bat` or run:
```bash
stop-all-trycloudflare.bat
```

This terminates all 4 services.

---

## How It Works

```
User's Browser (anywhere in the world)
         ↓
https://xyz-789.trycloudflare.com (Frontend Tunnel URL)
         ↓
Cloudflare Global Network
         ↓
TryCloudflare Tunnel (encrypted connection)
         ↓
Your PC - localhost:5173 (Frontend)
         ↓ (makes API calls)
https://abc-123.trycloudflare.com (Backend Tunnel URL)
         ↓
Cloudflare Global Network
         ↓
TryCloudflare Tunnel (encrypted connection)
         ↓
Your PC - localhost:8000 (Backend)
```

**Key Features:**
- ✅ Automatic HTTPS/SSL (no certificate warnings)
- ✅ Works from any device, anywhere
- ✅ No port forwarding needed
- ✅ No router configuration needed
- ✅ WebSocket support for real-time transcription
- ✅ Cloudflare's global CDN for fast access

---

## Troubleshooting

### Tunnels won't start

**Problem:** `cloudflared: command not found`

**Solution:** Make sure cloudflared is installed and in your PATH:
```bash
cloudflared --version
```

If not installed, download from: https://github.com/cloudflare/cloudflared/releases

---

### Can't access the application

**Problem:** Frontend loads but shows connection errors

**Solution:**
1. Make sure **all 4 services are running** (backend, frontend, both tunnels)
2. Verify you **updated `frontend/src/config.js`** with the Backend Tunnel URL
3. Make sure you used `https://` and `wss://` (not `http://` or `ws://`)
4. **Restart the frontend** after updating config.js

---

### Microphone not working

**Problem:** Browser blocks microphone access

**Solution:**
1. Make sure you're accessing via **HTTPS** (the Frontend Tunnel URL)
2. Never use `http://localhost:5173` when using tunnels
3. Check browser permissions (should show a microphone icon in address bar)
4. Try a different browser (Chrome/Edge recommended)

---

### WebSocket connection fails

**Problem:** Real-time transcription doesn't work

**Solution:**
1. Verify `WS_BASE_URL` in `frontend/src/config.js` uses `wss://` (not `ws://`)
2. Make sure you're using the **Backend Tunnel URL** (same as API_BASE_URL but with `wss://`)
3. Check that the backend is running and accepting WebSocket connections
4. Look at the Backend Tunnel window for error messages

---

### Frontend shows old API URL

**Problem:** Frontend still tries to connect to `localhost` or old tunnel URL

**Solution:**
1. Make sure you **saved** `frontend/src/config.js` after editing
2. **Hard refresh** the browser (Ctrl+Shift+R or Cmd+Shift+R)
3. **Restart the frontend dev server** to pick up the new configuration
4. Clear browser cache if needed

---

## TryCloudflare vs Named Tunnel Comparison

| Feature | TryCloudflare | Named Tunnel |
|---------|--------------|--------------|
| **Setup** | One command, instant | Requires login, config file, DNS |
| **Account** | None needed | Cloudflare account required |
| **URL** | Random (e.g., `abc-123.trycloudflare.com`) | Custom (e.g., `interview.yourdomain.com`) |
| **Persistence** | New URL each restart | Same URL forever |
| **Configuration** | No config files | Requires `config.yml` |
| **Multi-service** | Need 2 tunnels (frontend + backend) | 1 tunnel can route multiple services |
| **Uptime** | No SLA, testing only | Production-ready |
| **Best for** | Demos, testing, development | Production, permanent deployments |

---

## Advanced Tips

### Using a Custom Hostname (Optional)

If you want consistent URLs (but still no account needed), you can use a **subdomain**:

```bash
cloudflared tunnel --url http://localhost:8000 --hostname myapp.mydomain.com
```

**Requirements:**
- You must own `mydomain.com`
- You must add a CNAME record in your DNS pointing to the tunnel
- Not recommended for TryCloudflare (defeats the purpose of "instant")

### Viewing Tunnel Metrics

TryCloudflare tunnels expose metrics at:
```
http://localhost:20241/metrics
```

Visit this URL in your browser to see connection stats, request counts, etc.

### Running Tunnels in Background

To run tunnels without keeping terminal windows open:

**Windows:**
```bash
start /B cloudflared tunnel --url http://localhost:8000 > backend-tunnel.log 2>&1
start /B cloudflared tunnel --url http://localhost:5173 > frontend-tunnel.log 2>&1
```

Check `backend-tunnel.log` and `frontend-tunnel.log` for the URLs.

---

## Security Notes

- TryCloudflare tunnels are **publicly accessible** - anyone with the URL can access your app
- URLs are **random and hard to guess**, but not secret
- For production or sensitive data, use authentication in your application
- TryCloudflare has **no SLA** and may disconnect without warning
- Consider using a **named tunnel** with custom domain for production deployments

---

## Cost

- TryCloudflare: **Free** (no limits, no account needed)
- Bandwidth: **Unlimited**
- Number of tunnels: **Unlimited**
- Duration: **Unlimited** (tunnels stay active as long as the process runs)

---

## Additional Resources

- **TryCloudflare Docs:** https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/do-more-with-tunnels/trycloudflare/
- **Cloudflared GitHub:** https://github.com/cloudflare/cloudflared
- **Troubleshooting:** https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/troubleshooting/

---

## Support

If you encounter issues:

1. Check the tunnel windows for error messages
2. Verify all 4 services are running
3. Confirm you updated `frontend/src/config.js` correctly
4. Try restarting all services with `stop-all-trycloudflare.bat` then `start-all-trycloudflare.bat`
5. Check the troubleshooting section above

For project-specific issues, see the main `README.md` and `CLAUDE.md` files.
