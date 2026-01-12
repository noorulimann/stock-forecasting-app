# 🚀 Vercel MCP Setup Guide for VS Code with Copilot

## What is Vercel MCP?

Vercel MCP (Model Context Protocol) allows GitHub Copilot in VS Code to:
- ✅ Deploy your projects to Vercel directly
- ✅ Manage deployments
- ✅ Analyze deployment logs
- ✅ Search Vercel documentation

---

## Step-by-Step Setup

### **Step 1: Add Vercel MCP Server**

1. **Open Command Palette:**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)

2. **Run this command:**
   ```
   MCP: Add Server
   ```

3. **Select:** `HTTP`

4. **Enter these details:**
   ```
   URL: https://mcp.vercel.com
   Name: Vercel
   ```

5. **Select:** `Global` (to use across all projects)

6. **Click:** `Add`

---

### **Step 2: Start the Server and Authorize**

1. **Open Command Palette again:**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)

2. **Run this command:**
   ```
   MCP: List Servers
   ```

3. **Select:** `Vercel`

4. **Click:** `Start Server`

5. **When the dialog appears:**
   - "The MCP Server Definition 'Vercel' wants to authenticate to Vercel MCP"
   - **Click:** `Allow`

6. **When popup asks:**
   - "Do you want Code to open the external website?"
   - **Click:** `Cancel` (yes, cancel!)

7. **Next message appears:**
   - "Having trouble authenticating to 'Vercel MCP'? Would you like to try a different way? (URL Handler)"
   - **Click:** `Yes`

8. **Click:** `Open`

9. **Complete the Vercel sign-in flow** in your browser

10. **You're now connected!** ✅

---

### **Step 3: Verify Connection**

1. **Open Command Palette:**
   - Press `Ctrl+Shift+P`

2. **Run:**
   ```
   MCP: List Servers
   ```

3. **You should see:**
   - ✅ Vercel (Status: Connected)

---

### **Step 4: Use Vercel MCP with Copilot**

Now you can ask GitHub Copilot to:

**Example prompts:**

```
"Deploy this project to Vercel"

"Show me my Vercel deployments"

"Check the logs for my latest deployment"

"Create a new Vercel project for stock-forecasting-app"
```

---

## Project-Specific Setup (Recommended)

For better context and automatic project detection, add a project-specific MCP connection:

### **Find Your Team and Project Slugs:**

1. Go to: https://vercel.com/dashboard
2. Navigate to your project → Settings → General
3. Note your:
   - **Team Slug:** (your team name in the URL)
   - **Project Slug:** (your project name)

### **Add Project-Specific MCP:**

1. **Open Command Palette:** `Ctrl+Shift+P`
2. **Run:** `MCP: Add Server`
3. **Select:** `HTTP`
4. **Enter:**
   ```
   URL: https://mcp.vercel.com/YOUR_TEAM_SLUG/stock-forecasting-app-vercel
   Name: Vercel-StockApp
   ```
5. **Select:** `Workspace` (for this project only)
6. **Click:** `Add`
7. **Repeat Steps 2 from main setup to authorize**

---

## Quick Commands Reference

| Action | Command Palette |
|--------|----------------|
| Add Server | `MCP: Add Server` |
| List Servers | `MCP: List Servers` |
| Start Server | `MCP: Start Server` |
| Stop Server | `MCP: Stop Server` |
| Remove Server | `MCP: Remove Server` |

---

## Troubleshooting

### **"Command not found: MCP: Add Server"**

**Solution:** Update VS Code and GitHub Copilot extension:
1. Go to Extensions (`Ctrl+Shift+X`)
2. Search for "GitHub Copilot"
3. Click "Update" if available
4. Restart VS Code

### **"Authentication failed"**

**Solution:**
1. Open Command Palette
2. Run `MCP: Remove Server`
3. Select Vercel
4. Repeat setup from Step 1

### **"Server not responding"**

**Solution:**
1. Check internet connection
2. Verify URL is exactly: `https://mcp.vercel.com`
3. Try restarting VS Code

---

## What You Can Do After Setup

Once connected, you can ask Copilot to:

✅ **Deploy your project:**
- "Deploy this Flask app to Vercel"
- "Create a new Vercel deployment"

✅ **Manage deployments:**
- "Show my recent deployments"
- "Get deployment status for stock-forecasting-app"

✅ **Check logs:**
- "Show logs for the latest deployment"
- "What errors occurred in my last deployment?"

✅ **Manage projects:**
- "List all my Vercel projects"
- "Get project settings"

---

## Next Steps After Setup

1. ✅ Complete the MCP setup above
2. ✅ Ask Copilot: "Deploy stock-forecasting-app to Vercel"
3. ✅ Copilot will use the MCP to:
   - Create/update Vercel project
   - Deploy your code
   - Set environment variables
   - Provide deployment URL

---

## Alternative: Manual Setup

If you prefer to set up manually without MCP:

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to project
cd stock-forecasting-app

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

---

## Security Notes

⚠️ **Important:**
- Vercel MCP has the same access as your Vercel account
- Only connect to official endpoint: `https://mcp.vercel.com`
- Review changes before confirming deployments
- Enable human confirmation in workflows

---

## Support

- **Vercel MCP Docs:** https://vercel.com/docs/mcp/vercel-mcp
- **VS Code MCP:** Check Command Palette for MCP commands
- **Vercel Support:** https://vercel.com/help

---

**Status:** Ready to set up! Follow steps above. ✅

**Last Updated:** January 12, 2026
