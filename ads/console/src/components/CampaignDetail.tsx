import { Campaign, CampaignStats, pct, usd } from '../api';

/** Headline numbers for one campaign plus a budget bar (spend today / daily budget). */
export function CampaignDetail({ campaign, stats }: { campaign: Campaign; stats: CampaignStats }) {
  const ctr = stats.impressions > 0 ? stats.clicks / stats.impressions : 0;
  const used = Math.min(1, stats.spendMicros / campaign.dailyBudgetMicros);
  return (
    <section className="card detail" aria-labelledby="detail-title">
      <h2 id="detail-title">{campaign.adId}</h2>
      <dl className="metrics">
        <div><dt>Impressions</dt><dd>{stats.impressions.toLocaleString('en-US')}</dd></div>
        <div><dt>Clicks</dt><dd>{stats.clicks.toLocaleString('en-US')}</dd></div>
        <div><dt>CTR</dt><dd>{pct(ctr)}</dd></div>
        <div><dt>Predicted CTR</dt><dd>{pct(stats.pctr)}</dd></div>
        <div><dt>Spend</dt><dd>{usd(stats.spendMicros)}</dd></div>
      </dl>
      <div className="budget">
        <div className="budget-label">
          <span>Budget used today</span>
          <span>{usd(stats.spendMicros)} of {usd(campaign.dailyBudgetMicros)}</span>
        </div>
        <div className="bar" role="progressbar" aria-label="Budget used today" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={Math.round(used * 100)}>
          <div className="fill" style={{ width: `${used * 100}%` }} />
        </div>
      </div>
    </section>
  );
}
