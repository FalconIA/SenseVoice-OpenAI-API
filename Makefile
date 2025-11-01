init:
ifeq ($(OS),Windows_NT)
	@where uv > NUL 2> NUL || pip install uv -i https://mirrors.aliyun.com/pypi/simple
	set "UV_HTTP_TIMEOUT=300" && uv sync $(if $(VIRTUAL_ENV),--active,) --frozen --all-extras --no-install-project
else
	@which uv > /dev/null 2>&1 || pip install uv -i https://mirrors.aliyun.com/pypi/simple
	export UV_HTTP_TIMEOUT=300 && uv sync $(if $(VIRTUAL_ENV),--active,) --frozen --all-extras --no-install-project
endif

run:
	uv run $(if $(VIRTUAL_ENV),--active,) python -m src.main

build:
	uv build

fmt:
	@uv run $(if $(VIRTUAL_ENV),--active,) black ./src ./tests
	@uv run $(if $(VIRTUAL_ENV),--active,) isort --profile black ./src ./tests
	@$(MAKE) lint

lint:
	@uv run $(if $(VIRTUAL_ENV),--active,) pflake8 ./src ./tests

coverage: lint
	@uv run $(if $(VIRTUAL_ENV),--active,) pytest --cov=src tests
