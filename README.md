# Interactive Composition Explorer 🧊

ICE is a Python library and trace visualizer for language model programs.

## Screenshot

<p align="center">
  <img alt="ice-screenshot" src="https://user-images.githubusercontent.com/382515/192681645-6ed87072-2dc6-4982-92d1-8de209bc3ef6.png" />
  Execution trace visualized in ICE
</p>

## Features

- Run language model recipes in different modes: humans, human+LM, LM
- Inspect the execution traces in your browser for debugging
- Define and use new language model agents, e.g. chain-of-thought agents
- Run recipes quickly by parallelizing language model calls
- Reuse component recipes such as question-answering, ranking, and verification

## ICE is pre-1.0

:warning: **The ICE API may change at any point.** The ICE interface is being actively developed and we may change the API at any point, including removing functionality, renaming methods, splitting ICE into multiple projects, and other similarly disruptive changes. Use at your own risk.

## Getting started

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

1. Clone the repository: `git clone https://github.com/oughtinc/ice.git && cd ice`

1. Add required secrets to `.env`. See [`.env.example`](https://github.com/oughtinc/ice/blob/main/.env.example) for the format.

1. Start ICE in its own terminal and leave it running:

   ```sh
   scripts/run-local.sh
   ```

1. Start a `bash` session into the ICE container in another terminal:

   ```sh
   docker compose exec ice bash
   ```

1. Go through [the Primer](https://primer.ought.org/).

## Terminology

- **Recipes** are decompositions of a task into subtasks.

  The meaning of a recipe is: If a human executed these steps and did a good job at each workspace in isolation, the overall answer would be good. This decomposition may be informed by what we think ML can do at this point, but the recipe itself (as an abstraction) doesn’t know about specific agents.

- **Agents** perform atomic subtasks of predefined shapes, like completion, scoring, or classification.

  Agents don't know which recipe is calling them. Agents don’t maintain state between subtasks. Agents generally try to complete all subtasks they're asked to complete (however badly), but some will not have implementations for certain task types.

- The **mode** in which a recipe runs is a global setting that can affect every agent call. For instance, whether to use humans or agents. Recipes can also run with certain `RecipeSettings`, which can map a task type to a specific `agent_name`, which can modify which agent is used for that specfic type of task.

## Additional resources

1. [Join the ICE Slack channel](https://join.slack.com/t/ice-1mh7029/shared_invite/zt-1h8118i28-tPDSulG8C~4dr5ZdAky1gg) to collaborate with other people composing language model tasks. You can also use it to ask questions about using ICE.

2. [Watch the recording of Ought's Lab Meeting](https://www.youtube.com/watch?v=cZqq4muY5_w) to understand the high-level goals for ICE, how it interacts with Ought's other work, and how it contributes to alignment research.

3. [Read the ICE announcement post](https://ought.org/updates/2022-10-06-ice-primer) for another introduction.

## Contributions

ICE is an [open-source](https://github.com/oughtinc/ice/blob/main/LICENSE.md) project by [Ought](https://ought.org/). We're an applied ML lab building the AI research assistant [Elicit](https://elicit.org/).

We welcome community contributions:

- If you're a developer, you can dive into the codebase and help us fix bugs, improve code quality and performance, or add new features.
- If you're a language model researcher, you can help us add new agents or improve existing ones, and refine or create new recipes and recipe components.

For larger contributions, make an issue for discussion before submitting a PR.

And for even larger contributions, join us - [we're hiring](https://ought.org/careers)!

### Releases

To release a new version of ICE, follow these steps:

1. Update the version number in:

   - `docker-compose*.yml`
   - `pyproject.toml`
   - `package.json`
   - `scripts/run-local.sh`

1. Regenerate the `poetry.lock` file:

   ```sh
   docker compose exec ice poetry lock --no-update
   ```

1. Regenerate the `package-lock.json` file:

   ```sh
   docker compose exec ice npm --prefix ui install --package-lock-only
   ```

1. Update `CHANGELOG.md`.

1. Commit the changes.

1. Tag the commit with the version number:

   ```sh
   git tag <version>
   ```

1. Open a PR and verify that CI passes.

1. Build and push the Docker images:

   ```sh
   # TODO: Script this, sharing code with scripts/run-local.sh.
   docker buildx bake -f docker-compose.yml -f docker-compose.build.yml --push
   docker buildx bake -f docker-compose.yml -f docker-compose.streamlit.yml -f docker-compose.build-streamlit.yml --push
   docker buildx bake -f docker-compose.yml -f docker-compose.torch.yml -f docker-compose.build-torch.yml --push
   ```

1. Push the tag:

   ```sh
   git push --tags
   ```

1. Merge the PR.
