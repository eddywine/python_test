# python_test
---
kind: adr
number: 12
title: Proposed ETL Pipeline
author: edwin@kwara.com
---

- **Date:** 18th October 2020
- **Status:** Proposed

## Context

We're now accepting user signups from the onboarding Wizard to capture leads and to enable people to try out Kwara for a defined period of time.

Once these users have signed up, they'll need to be able to log in into Kwara
using the existing login page on the WebApp

At the time of this writing, we store _active_ users on Mambu and the _prospective_ users on Kwara and thus have two different login flows & endpoints to facilitate this; which is not ideal.

The goal is to unify these two login endpoints into one and handle the account logic in the backend. This is should work because both endpoints accept similar credentials.

```
json
{
  "auth": {
    "username": "<the username/email >",
    "password": "<user password>"
  }
}
```

## Decision

- [ ] For prospective users, create an account on Kwara. Upon conversion, subsequently create a dedicated mambu account for them reusing the same credentials.
- [ ] For existing "SACCO Users", reuse their existing credentials to _gracefully_ create an account on Kwara if none exists upon login. This should make the transition seamless.
- [ ] Merge the new `POST /login` endpoint into the existing `POST /auth/user_token` endpoint retaining the current logic on the WebApp
- [ ] Ensure users who join Kwara via both invitations and signup have an account with Kwara

Auth Flow Diagram:

![Copy of Unified Login Flow (2)](https://user-images.githubusercontent.com/17295175/56960144-8722a700-6b58-11e9-9863-03af94984c2d.jpg)

## Consequences

- We retain current benefits to having SACCO users partially delegated to Mambu such as permissions/roles handling
- We take back **some** ownership of Mambu users, if we were ever to swap out Mambu, we'd still own the user accounts and would migrate gracefully.
- We will have duplicate accounts for _active_ users, one on Kwara associated with one on Mambu. The added overhead here is that we have to ensure they're always in sync whenever account details are changed. E.g. password resets.
