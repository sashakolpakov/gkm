"""Standalone GKM campaign machinery for the RoboArm environment.

Nothing in this package imports or adapts an ARC runtime. The public proposer
surface is materialized into a proposal-only clean workspace with no connector
handle. The authoritative simulator remains behind a host-owned deterministic
safety FSA and a single-use commit permit.
"""

from .runner import CampaignConfig, CampaignResult, run_campaign

__all__ = ["CampaignConfig", "CampaignResult", "run_campaign"]
