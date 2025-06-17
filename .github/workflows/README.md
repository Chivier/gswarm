# GitHub Workflows for gswarm

This repository includes several GitHub Actions workflows to automate testing, building, and publishing.

## Workflows


### 1. PyPI Publishing Workflow (`.github/workflows/publish-to-pypi.yml`)

Publishes the package to PyPI when a new release is created.

**Automatic publishing:**
- Triggers when a GitHub release is published
- Builds the package using `hatchling`
- Tests installation on multiple Python versions
- Publishes to PyPI using trusted publishing

**Manual publishing:**
- Can be triggered manually from the Actions tab
- Option to publish to Test PyPI for testing


## Setup Instructions

### 1. Configure PyPI Trusted Publishing

Before the publishing workflow can work, you need to configure trusted publishing on PyPI:

1. Go to [PyPI](https://pypi.org) and log in to your account
2. Navigate to "Publishing" → "Add a new pending publisher"
3. Fill in the details:
   - **PyPI Project Name**: `gswarm`
   - **Owner**: Your GitHub username/organization
   - **Repository name**: `gswarm-profiler` (or your repo name)
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `pypi`

4. For Test PyPI, repeat the process at [test.pypi.org](https://test.pypi.org) with environment name `testpypi`

### 2. Create GitHub Environments

1. Go to your repository → Settings → Environments
2. Create two environments:
   - `pypi` (for production PyPI)
   - `testpypi` (for Test PyPI)
3. Add protection rules if desired (e.g., require reviews)

### 3. Publishing a Release

To publish a new version:

1. Update the version in `pyproject.toml`
2. Commit and push the changes
3. Create a new release on GitHub:
   - Go to Releases → Create a new release
   - Create a new tag (e.g., `v0.5.0`)
   - Add release notes
   - Click "Publish release"

The workflow will automatically build and publish to PyPI.

### 4. Manual Publishing (Testing)

To test publishing without creating a release:

1. Go to Actions → "Publish to PyPI"
2. Click "Run workflow"
3. Check "Publish to Test PyPI instead of PyPI"
4. Click "Run workflow"

This will publish to Test PyPI for testing purposes.

## Troubleshooting

### Common Issues

1. **Publishing fails with authentication error:**
   - Ensure trusted publishing is set up correctly on PyPI
   - Check that the environment names match exactly

2. **Tests fail:**
   - Check the CI workflow logs for specific error details
   - Ensure all dependencies are properly specified in `pyproject.toml`

3. **Version conflicts:**
   - Make sure to update the version in `pyproject.toml` before releasing
   - PyPI doesn't allow re-uploading the same version

### Security Notes

- These workflows use GitHub's OIDC trusted publishing, which is more secure than API tokens
- No sensitive credentials are stored in the repository
- The workflows only run in your repository and can't be triggered by forks

## Customization

You can customize these workflows by:

- Modifying the Python versions in the test matrix
- Adding or removing code quality checks
- Changing the schedule for dependency updates
- Adding additional environments or deployment targets
