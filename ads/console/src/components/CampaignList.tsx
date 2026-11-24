import { Campaign, usd } from '../api';

interface Props {
  campaigns: Campaign[];
  selected: string | null;
  onSelect: (adId: string) => void;
  showAdvertiser: boolean;
}

/** A table on wide screens; the same rows stack into cards under 640px (see styles.css). */
export function CampaignList({ campaigns, selected, onSelect, showAdvertiser }: Props) {
  if (campaigns.length === 0) return <p className="muted">No campaigns yet.</p>;
  return (
    <table className="campaigns">
      <caption className="sr-only">Your campaigns</caption>
      <thead>
        <tr>
          <th scope="col">Ad</th>
          {showAdvertiser && <th scope="col">Advertiser</th>}
          <th scope="col">Job</th>
          <th scope="col">Max CPC</th>
          <th scope="col">Daily budget</th>
        </tr>
      </thead>
      <tbody>
        {campaigns.map((c) => (
          <tr key={c.adId} className={c.adId === selected ? 'selected' : undefined}>
            <td data-label="Ad">
              <button className="link" onClick={() => onSelect(c.adId)} aria-pressed={c.adId === selected}>
                {c.adId}
              </button>
            </td>
            {showAdvertiser && <td data-label="Advertiser">{c.advertiserId}</td>}
            <td data-label="Job">#{c.jobId}</td>
            <td data-label="Max CPC">{usd(c.bidMicros)}</td>
            <td data-label="Daily budget">{usd(c.dailyBudgetMicros)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
