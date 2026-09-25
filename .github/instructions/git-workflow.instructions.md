---
description: "Require explicit user approval before publishing repository changes."
applyTo: "**"
---

- Never run `git push` or otherwise publish changes to a remote unless the user explicitly requests that action in the current conversation.
- A previous request to push does not authorize later pushes; ask for or wait for fresh explicit approval for each push.
- Do not create pull requests, publish releases, or mutate remote branches without explicit current-conversation authorization.
- Local commits, branches, and validation are allowed unless the user asks otherwise.
