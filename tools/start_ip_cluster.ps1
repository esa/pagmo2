# Small script to start ipcluster as a background process.

# Start-Process "ipcluster" -ArgumentList "start"
Start-Job -ScriptBlock {ipcluster start}
