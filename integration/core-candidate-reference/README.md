# Core prerequisite reference, not an applied main-repo change

This candidate runtime patch was compared against minh2509/autobiz deployed SHA b64e41aedb121d41b39adfcc6fbfb352d400c5da during the earlier read-only audit. It includes customer ingress/worker/approval/result wiring tested in AutoBiz_Core_Integration. It has NOT been transferred to autobiz-main or deployed.

The integrator must compare it with the current full main repository and current deployed revision, resolve differences and run Core CI before rollout. Do not blindly apply it to a different HEAD. No migrations, credentials or production permissions are supplied by this patch. Core/Owner integration remains the recipient's next step.
