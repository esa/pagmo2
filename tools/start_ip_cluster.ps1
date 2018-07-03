# Small script to start ipcluster as a background process.

# NOTE: the Start-Job command seems to work well on appveyor. Earlier on
# I tried also with Start-Process but it would get stuck on Python 3.6.

# Start-Process "ipcluster" -ArgumentList "start"

Start-Job -ScriptBlock {ipcluster start}
