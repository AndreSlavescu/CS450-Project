EXCLUDE := -not -path '*/ThunderKittens/*'

FIND_PY   = find . -type f -name '*.py' $(EXCLUDE)
FIND_CUDA = find . -type f \( -name '*.cuh' -o -name '*.cu' -o -name '*.h' -o -name '*.cpp' \) $(EXCLUDE)

CPPLINT_FILTERS := -legal/copyright,-build/header_guard,-build/include_subdir,-runtime/references,-build/namespaces

.PHONY: lint format lint-python lint-cpp format-python format-cpp

lint:
	@$(MAKE) --no-print-directory -j2 lint-python lint-cpp

format:
	@$(MAKE) --no-print-directory -j2 format-python format-cpp

lint-python:
	$(FIND_PY) | xargs ruff check

format-python:
	$(FIND_PY) | xargs black

lint-cpp:
	$(FIND_CUDA) | xargs cpplint --filter=$(CPPLINT_FILTERS)

format-cpp:
	$(FIND_CUDA) | xargs clang-format -i
