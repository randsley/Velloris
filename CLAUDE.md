# Claude Code Guidelines for Velloris

## iOS/Xcode Conventions

When adding new Swift or Objective-C files to the project:
1. Create the source files in their appropriate directories
2. **Always update the Xcode `.pbxproj` file** to include the new file references and build file entries
3. Ensure the file is added to the correct target and build phase
4. After adding files, run `xcodebuild` to verify the project compiles without errors
5. Do not declare the task complete until the build succeeds

This prevents the common pitfall of creating source files that exist on disk but aren't referenced in the project file, causing build failures.

## Information Lookup

When asked to find, review, or reference project plans, documentation, specifications, or design notes:
1. **Always search markdown files first** using Glob patterns (`*.md`) and the Read tool
2. Check the following locations in order:
   - Project root (README.md, CLAUDE.md, design docs)
   - `/docs` directory (if it exists)
   - Any top-level `.md` files in the repository
3. Look for files containing keywords like "plan", "spec", "design", "roadmap", "architecture", or "todo"
4. Only use task list tools or other sources if markdown files don't contain the relevant information
5. Summarize what you found and cite the exact file and section

This ensures documentation is consulted from the actual source of truth in the repository.

## Testing & Validation

After making code changes:

**For Python scripts:**
- Always perform a quick syntax check: `python -c 'import ast; ast.parse(open("path/to/file.py").read())'`
- Or run `python -m py_compile path/to/file.py` to verify the file is syntactically valid
- For executable scripts, do a dry-run or `--help` check before declaring success
- Do not stop after fixing one error—trace through the entire execution path for cascading issues

**For Swift/Xcode projects:**
- After making changes, run `xcodebuild clean build` to verify compilation
- Check for any compiler warnings or errors
- If tests exist, run the test suite before declaring completion
- Verify that all modified files are included in the project file

Do not declare a fix complete until the code has been validated by running it or building it. Catching errors during this validation step prevents cascading failures in subsequent sessions.

## General Principles

- When executing Python scripts with issues, provide full error output upfront so all errors can be addressed in fewer iterations
- For multi-file changes, start with a checklist of all files that need to be created or modified, including project configuration files
- Iterate autonomously through fix cycles without stopping after each error—read the next error and continue fixing until the execution is clean
