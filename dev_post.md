---
title: Haunted Hardware: a séance for your PC, run by open models
published: false
tags: devchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

## What I Built

<!-- TODO: 3-5 sentences in YOUR voice. Hook first. Why a haunted computer? What did your machine do that made you think "something is possessing this thing"? -->

Haunted Hardware watches your computer's vitals (CPU, RAM, temperature, the busiest process) and decides whether the machine is "possessed". It names the entity, from a Memory Leak Specter to a CPU Poltergeist or a Thermal Wraith, and a local open-weight model writes an exorcism log that ends with a real, plain-English suggestion for what to do about it.

To be clear about what it is: a fun anomaly detector, not antivirus. The classifier is trained on synthetic data I generated, and the "ghosts" are categories of unusual behaviour.

## Demo

<!-- TODO: embed your video (YouTube/Loom) and link the hosted demo -->
- Video: LINK
- Hosted demo (simulated data, see note below): LINK
- The hosted demo can't see *your* machine, since a website can't read your processes. It runs on simulated readings and manual sliders. The real tool runs locally.

## Code

<!-- TODO: embed your repo: {% embed https://github.com/YOUR_USER/haunted-hardware %} -->

## How I Built It

- **psutil** samples CPU, RAM, temperature and the busiest process.
- **TabPFN** (Prior Labs' open-weight tabular foundation model) classifies the readings into Calm or one of three entities, and the possession % is the probability of "not calm".
  <!-- TODO: say honestly how you trained it (synthetic ranges, how many rows) and show one result, e.g. a screenshot of the Poltergeist being caught -->
- **Gemma, run locally through Ollama**, writes the exorcism log from the readings.
- **FastAPI + one HTML page** for the dashboard, with Live, Simulate and Manual-slider modes.
- Deployed the demo on **Render** (DEMO_MODE) <!-- TODO: and/or your .tech domain; delete whichever you didn't use -->

## Why Does Open Innovation Matter?

<!-- TODO: rewrite in your own words, this section is what the prompt asks for. Seed ideas: -->
A tool that inspects your running processes, file paths and network activity is exactly the kind of data you shouldn't ship to someone else's API. Because Gemma runs through Ollama on my own machine, nothing about my system leaves it, it works with the Wi-Fi off, and there's no API key to leak or bill to run up.
It's also swappable: one environment variable (`HAUNT_MODEL`) changes the model, and I could point it at a bigger Gemma on a GPU box without touching the code.
TabPFN being open-weight meant I could run the classifier locally too, with no training loop and no cloud call. <!-- TODO: add one honest limitation you hit, e.g. setup friction, speed on CPU, the small model's writing quality -->

## The Spirit of It

<!-- TODO: the moment it landed. Did it catch a real runaway process on your laptop? Did the ghost's line make you laugh? Show it. If nothing real happened, delete this section. -->

## Prize Categories

<!-- Only list what you genuinely used and can show: -->
- Best Use of TabPFN
- Best Use of Gemma
- Best Use of Render
- Best Use of a .tech Domain (only if live on one)
