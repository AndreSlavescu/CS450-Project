SUBMODULES := $(shell git config --file .gitmodules --get-regexp path | awk '{print $$2}')
EXCLUDE    := $(foreach sm,$(SUBMODULES),-not -path './$(sm)/*')

FIND_PY   = find . -type f -name '*.py' $(EXCLUDE)
FIND_CUDA = find . -type f \( -name '*.cuh' -o -name '*.cu' -o -name '*.h' -o -name '*.cpp' \) $(EXCLUDE)

CPPLINT_FILTERS := -legal/copyright,-build/header_guard,-build/include_subdir,-runtime/references,-build/namespaces,-whitespace/line_length,-runtime/int,-build/include_what_you_use,-whitespace/comments

.PHONY: lint format lint-python lint-cpp format-python format-cpp

lint:
	@$(MAKE) --no-print-directory -j2 lint-python lint-cpp

format:
	@$(MAKE) --no-print-directory -j2 format-python format-cpp

lint-python:
	$(FIND_PY) | xargs ruff check --fix

format-python:
	$(FIND_PY) | xargs black

lint-cpp:
	$(FIND_CUDA) | xargs cpplint --filter=$(CPPLINT_FILTERS)

format-cpp:
	$(FIND_CUDA) | xargs clang-format -i
