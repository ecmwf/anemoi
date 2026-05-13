.. _contributing:

##############
 Contributing
##############

Thank you for your interest in Anemoi! This guide will show you how to
contribute to the Anemoi packages.

****************
 Raise an issue
****************

If you encounter a bug or have a feature request, the first step is to
let us know by raising an issue on GitHub using the following steps:

#. Check the existing issues to avoid duplicates.

#. If it's a new issue, create a detailed bug report or feature request
   by filling in the issue template.

#. Use clear, descriptive titles and provide as much relevant
   information as possible.

#. If you have a bug, include the steps to reproduce it.

#. If you have a feature request, describe the use case and expected
   behaviour.

#. If you are interested in solving the issue yourself, assign the issue
   to yourself and follow the steps below.

#. Tag your issue with the appropriate labels to help the community
   identify and triage it effectively. Be sure to follow the
   :ref:`labelling-guidelines`.

**********************
 Developing in Anemoi
**********************

For contributing to the development of the Anemoi packages, please
follow these steps:

#. Fork the anemoi repository on GitHub to your personal/organisation
   account. See the `GitHub tutorial
   <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

#. Set up the development environment following the instructions in the
   :ref:`setting-up-the-development-environment` section.

#. Create a new branch for your developments, following the
   :ref:`branching-guidelines`.

#. Make your changes and ensure that your changes adhere to the
   :ref:`development-guidelines`.

#. Commit your changes and push your branch to your fork on GitHub.

#. Open a Pull Request against the `main` branch of the original
   repository, set a PR title according to the
   :ref:`pr-title-guidelines` and fill in the Pull Request template.

#. Sign the Contributor License Agreement (CLA) on GitHub. If you
   haven't yet signed the CLA for the repository, you will be prompted
   to do so in a comment on your pull request.

#. Request a review from maintainers or other contributors, which will
   follow the :ref:`code-review-process`.

.. _code-review-process:

*********************
 Code Review Process
*********************

The Anemoi packages have a set of automated checks to enforce coding
guidelines. These checks are run via GitHub Actions on every Pull
Request. For security reasons, maintainers must review code changes
before enabling automated checks.

#. Ensure that all the :ref:`development-guidelines` criteria are met
   before submitting a Pull Request, and pay special attention to the
   :ref:`labelling-guidelines`.

#. Request a review from maintainers or other contributors, noting that
   support is on a best-efforts basis.

#. After an initial review, a maintainer will enable automated checks to
   run on the Pull Request.

#. It is the sole responsibility of the respective contributor to keep the PR
   in sync with the target branch (e.g., :code:`git pull origin main`)
   and to address any merge conflicts that may arise.

#. Reviewers may leave feedback or request changes. To confirm that
   feedback has been addressed, ask reviewers to mark their comments as
   resolved.

#. Once approved, the Pull Request will be merged into the appropriate
   branch according to the :ref:`merging-guidelines`

.. _inactive-issue-process:

*************************
 Inactive Issue Process
*************************

To keep our issue tracker focused on active work, we use an automated stale bot that marks and closes inactive issues and pull requests.


How It Works
------------

- **Issues**: After 90 days of inactivity, labeled as ``inactive``; closed after 10 more days
- **PRs**: After 90 days of inactivity, labeled as ``inactive``; closed after 10 more days

If during the grace period of 10 days, a maintainer or contributor provides an update on the issue or PR to indicate that it is still being worked on, the timer is reset and the issue or PR is not closed.

Exempt Labels
-------------

Some issues and PRs never go inactive:

**Issues:**

- ``good first issue`` - Newcomer-friendly issues
- ``help wanted`` - Issues seeking contributions
- ``roadmap`` - Planned features
- ``known-issue`` - Confirmed issues not yet fixed

If your issue is still actively being worked on but may take longer to resolve, you can apply the known-issue label to indicate that it should not be treated as inactive.
We ask contributors to use this mechanism in good faith and to not arbitrarily assign labels to circumvent the bot.

**PRs:**

- ``autorelease: pending`` - Release automation PRs

Preventing Stale Status
-----------------------

To keep your issue or PR active:

1. Add a comment - any activity resets the timer
2. Request an exempt label from a maintainer
3. Provide status updates

If Closed by Mistake
--------------------
If your issue was closed but is still relevant, simply comment to reopen it or create a new issue. Contact maintainers if you believe the closure was incorrect.


*****************
 Code of conduct
*****************

Please follow the `GitHub Code of Conduct
<https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct>`_
for respectful collaboration.
