<?php
// Set page title and include header
$title = "Sector Rotation Analysis";
include('header.php');

// Helper functions
function calculate_date_range($period_days)
{
    $end_date = new DateTime();
    $start_date = clone $end_date;
    $start_date->sub(new DateInterval('P' . ($period_days + 10) . 'D'));
    return [$start_date->format('Y-m-d'), $end_date->format('Y-m-d')];
}

function fetch_yahoo_data($symbol, $start_date, $end_date)
{
    $url = "https://query1.finance.yahoo.com/v8/finance/chart/$symbol?interval=1d&period1=" .
        strtotime($start_date) . "&period2=" . strtotime($end_date);

    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    $response = curl_exec($ch);
    curl_close($ch);

    $data = json_decode($response, true);

    if (!isset($data['chart']['result'][0]['indicators']['quote'][0]['close'])) {
        return null;
    }

    $prices = $data['chart']['result'][0]['indicators']['quote'][0]['close'];
    $timestamps = $data['chart']['result'][0]['timestamp'];

    $result = [];
    foreach ($timestamps as $i => $timestamp) {
        if ($prices[$i] !== null) {
            $result[date('Y-m-d', $timestamp)] = $prices[$i];
        }
    }

    return $result;
}

function calculate_relative_strength($sector_prices, $benchmark_prices, $period)
{
    // Align dates
    $aligned = [];
    foreach ($sector_prices as $date => $price) {
        if (isset($benchmark_prices[$date])) {
            $aligned[$date] = [
                'sector' => $price,
                'benchmark' => $benchmark_prices[$date]
            ];
        }
    }

    if (count($aligned) < $period) {
        return [null, null];
    }

    // Calculate relative strength
    $relative_strength = [];
    foreach ($aligned as $date => $prices) {
        $relative_strength[$date] = $prices['sector'] / $prices['benchmark'];
    }

    // Calculate momentum
    $dates = array_keys($relative_strength);
    $rs_values = array_values($relative_strength);
    $momentum = [];

    for ($i = $period; $i < count($rs_values); $i++) {
        $momentum[$dates[$i]] = ($rs_values[$i] / $rs_values[$i - $period] - 1) * 100;
    }

    return [$relative_strength, $momentum];
}

function calculate_jdk_rs_ratio($relative_strength, $long_period = 40)
{
    if (count($relative_strength) < $long_period) {
        return null;
    }

    $rs_values = array_values($relative_strength);
    $dates = array_keys($relative_strength);
    $rs_ratio = [];

    for ($i = $long_period; $i < count($rs_values); $i++) {
        $window = array_slice($rs_values, $i - $long_period, $long_period);
        $mean = array_sum($window) / $long_period;
        $rs_ratio[$dates[$i]] = ($rs_values[$i] / $mean) * 100;
    }

    return $rs_ratio;
}

function calculate_jdk_rs_momentum($rs_ratio, $period = 10)
{
    if (count($rs_ratio) < $period) {
        return null;
    }

    $rs_values = array_values($rs_ratio);
    $dates = array_keys($rs_ratio);
    $momentum = [];

    for ($i = $period; $i < count($rs_values); $i++) {
        $momentum[$dates[$i]] = (($rs_values[$i] / $rs_values[$i - $period]) - 1) * 100;
    }

    return $momentum;
}

function get_quadrant_info($rs_ratio, $rs_momentum)
{
    if ($rs_ratio > 100 && $rs_momentum > 0) {
        return ["Leading", "green", "🚀"];
    } elseif ($rs_ratio > 100 && $rs_momentum < 0) {
        return ["Weakening", "orange", "📉"];
    } elseif ($rs_ratio < 100 && $rs_momentum < 0) {
        return ["Lagging", "red", "📊"];
    } else {
        return ["Improving", "blue", "📈"];
    }
}

// Process form submission
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $benchmark = $_POST['benchmark'] ?? '^NSEI';
    $period = intval($_POST['period'] ?? 90);
    $tail_length = intval($_POST['tail_length'] ?? 4);
    $show_tail = isset($_POST['show_tail']);
    $sectors = array_filter(array_map('trim', explode("\n", $_POST['sectors'] ?? '')));

    list($start_date, $end_date) = calculate_date_range($period);

    // Fetch benchmark data
    $benchmark_data = fetch_yahoo_data($benchmark, $start_date, $end_date);

    if (!$benchmark_data) {
        $error = "Could not fetch data for benchmark: $benchmark";
    } else {
        // Fetch sector data
        $sector_data = [];
        $failed_sectors = [];

        foreach ($sectors as $sector) {
            $data = fetch_yahoo_data($sector, $start_date, $end_date);
            if ($data && count($data) >= 20) {
                $sector_data[$sector] = $data;
            } else {
                $failed_sectors[] = $sector;
            }
        }

        if (empty($sector_data)) {
            $error = "Could not fetch data for any sectors";
        } else {
            // Calculate metrics
            $results = [];

            foreach ($sector_data as $symbol => $prices) {
                list($rel_strength, $rel_momentum) = calculate_relative_strength($prices, $benchmark_data, 10);

                if ($rel_strength) {
                    $rs_ratio = calculate_jdk_rs_ratio($rel_strength);
                    $rs_momentum = calculate_jdk_rs_momentum($rs_ratio);

                    if ($rs_ratio && $rs_momentum) {
                        $results[$symbol] = [
                            'rs_ratio' => $rs_ratio,
                            'rs_momentum' => $rs_momentum,
                            'relative_strength' => $rel_strength
                        ];
                    }
                }
            }

            if (empty($results)) {
                $error = "Could not calculate metrics for any sectors";
            } else {
                // Prepare data for chart
                $chart_data = [];
                $summary_data = [];

                foreach ($results as $symbol => $data) {
                    $rs_ratio = $data['rs_ratio'];
                    $rs_momentum = $data['rs_momentum'];

                    // Get tail points
                    $ratio_values = array_values($rs_ratio);
                    $momentum_values = array_values($rs_momentum);
                    $dates = array_keys($rs_ratio);

                    $tail_points = min($tail_length, count($ratio_values));
                    $tail_start = count($ratio_values) - $tail_points;

                    $tail_ratio = array_slice($ratio_values, $tail_start);
                    $tail_momentum = array_slice($momentum_values, $tail_start);
                    $tail_dates = array_slice($dates, $tail_start);

                    // Current values
                    $current_ratio = end($ratio_values);
                    $current_momentum = end($momentum_values);
                    list($quadrant, $color, $icon) = get_quadrant_info($current_ratio, $current_momentum);

                    // Add to chart data
                    $chart_data[] = [
                        'symbol' => $symbol,
                        'ratio' => $current_ratio,
                        'momentum' => $current_momentum,
                        'color' => $color,
                        'tail_ratio' => $tail_ratio,
                        'tail_momentum' => $tail_momentum,
                        'quadrant' => $quadrant
                    ];

                    // Add to summary
                    $summary_data[] = [
                        'symbol' => $symbol,
                        'ratio' => number_format($current_ratio, 2),
                        'momentum' => number_format($current_momentum, 2),
                        'quadrant' => $quadrant
                    ];
                }
            }
        }
    }
}
?>

<div class="container">
    <h1>Sector Rotation - Relative Rotation Graph</h1>
    <p>Analyze sector/stock performance relative to benchmark using RRG methodology. Input Symbols as seen in Yahoo Finance</p>

    <?php if (isset($error)): ?>
        <div class="alert alert-danger"><?= htmlspecialchars($error) ?></div>
    <?php endif; ?>

    <form method="post">
        <div class="row">
            <div class="col-md-6">
                <div class="form-group">
                    <label for="benchmark">Benchmark Symbol</label>
                    <input type="text" class="form-control" id="benchmark" name="benchmark" value="<?= htmlspecialchars($_POST['benchmark'] ?? '^NSEI') ?>" placeholder="e.g., ^NSEI for Nifty 50">
                </div>

                <div class="form-group">
                    <label for="period">Analysis Period (days)</label>
                    <input type="range" class="form-control-range" id="period" name="period" min="65" max="365" value="<?= htmlspecialchars($_POST['period'] ?? 90) ?>" step="5">
                    <span id="periodValue"><?= htmlspecialchars($_POST['period'] ?? 90) ?></span> days
                </div>

                <div class="form-group">
                    <label for="tail_length">Tail Length (days)</label>
                    <input type="range" class="form-control-range" id="tail_length" name="tail_length" min="2" max="25" value="<?= htmlspecialchars($_POST['tail_length'] ?? 4) ?>" step="1">
                    <span id="tailLengthValue"><?= htmlspecialchars($_POST['tail_length'] ?? 4) ?></span> days
                </div>
            </div>

            <div class="col-md-6">
                <div class="form-group">
                    <label for="sectors">Enter Sector/Stock symbols (one per line)</label>
                    <textarea class="form-control" id="sectors" name="sectors" rows="8" placeholder="Enter each symbol on a new line"><?= htmlspecialchars($_POST['sectors'] ?? implode("\n", ["^CNXAUTO", "^CNXPHARMA", "^CNXMETAL", "^CNXIT", "^CNXENERGY", "^CNXREALTY", "^CNXPSUBANK", "^CNXMEDIA", "^CNXINFRA", "^CNXPSE", "RELIANCE.NS", "INFY.NS"])) ?></textarea>
                </div>

                <div class="form-check">
                    <input type="checkbox" class="form-check-input" id="show_tail" name="show_tail" <?= isset($_POST['show_tail']) ? 'checked' : '' ?>>
                    <label class="form-check-label" for="show_tail">Show Tail</label>
                </div>
            </div>
        </div>

        <button type="submit" class="btn btn-primary">Run Analysis</button>
    </form>

    <?php if (isset($chart_data)): ?>
        <div class="mt-4">
            <h2>Relative Rotation Graph</h2>
            <div id="rrgChart" style="width: 100%; height: 800px;"></div>
        </div>

        <div class="mt-4">
            <h2>Relative Positions of Sector/Stock</h2>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Sector</th>
                        <th>RS-Ratio</th>
                        <th>RS-Momentum</th>
                        <th>Quadrant</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($summary_data as $row): ?>
                        <tr>
                            <td><?= htmlspecialchars($row['symbol']) ?></td>
                            <td><?= htmlspecialchars($row['ratio']) ?></td>
                            <td><?= htmlspecialchars($row['momentum']) ?></td>
                            <td><?= htmlspecialchars($row['quadrant']) ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>

        <div class="mt-4">
            <h2>Understanding the Relative Rotation Graph</h2>
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Quadrants Explanation:</h5>
                    <ul>
                        <li><strong>Leading (Top-Right)</strong>: High relative strength, positive momentum
                            <ul>
                                <li>Sectors outperforming benchmark with increasing momentum</li>
                            </ul>
                        </li>
                        <li><strong>Weakening (Bottom-Right)</strong>: High relative strength, negative momentum
                            <ul>
                                <li>Sectors still outperforming but losing momentum</li>
                            </ul>
                        </li>
                        <li><strong>Lagging (Bottom-Left)</strong>: Low relative strength, negative momentum
                            <ul>
                                <li>Sectors underperforming benchmark with decreasing momentum</li>
                            </ul>
                        </li>
                        <li><strong>Improving (Top-Left)</strong>: Low relative strength, positive momentum
                            <ul>
                                <li>Sectors underperforming but gaining momentum</li>
                            </ul>
                        </li>
                    </ul>

                    <h5 class="card-title mt-3">How to Read:</h5>
                    <ul>
                        <li><strong>RS-Ratio > 100</strong>: Sector outperforming benchmark</li>
                        <li><strong>RS-Ratio < 100</strong>: Sector underperforming benchmark</li>
                        <li><strong>RS-Momentum > 0</strong>: Relative strength is improving</li>
                        <li><strong>RS-Momentum < 0</strong>: Relative strength is declining</li>
                        <li><strong>Tail</strong>: Shows the trajectory of sector movement over time</li>
                    </ul>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Update slider values
                document.getElementById('period').addEventListener('input', function() {
                    document.getElementById('periodValue').textContent = this.value;
                });

                document.getElementById('tail_length').addEventListener('input', function() {
                    document.getElementById('tailLengthValue').textContent = this.value;
                });

                // Chart data from PHP
                const chartData = <?= json_encode($chart_data) ?>;
                const showTail = <?= isset($_POST['show_tail']) ? 'true' : 'false' ?>;

                // Create RRG chart
                const ctx = document.getElementById('rrgChart').getContext('2d');

                // Determine chart boundaries
                const ratios = chartData.map(item => item.ratio);
                const momentums = chartData.map(item => item.momentum);

                const minRatio = Math.min(...ratios);
                const maxRatio = Math.max(...ratios);
                const minMomentum = Math.min(...momentums);
                const maxMomentum = Math.max(...momentums);

                const xPadding = (maxRatio - minRatio) * 0.1;
                const yPadding = (maxMomentum - minMomentum) * 0.1;

                const xRange = [minRatio - xPadding, maxRatio + xPadding];
                const yRange = [minMomentum - yPadding, maxMomentum + yPadding];

                // Create datasets
                const datasets = chartData.map(item => {
                    const dataset = {
                        label: item.symbol,
                        data: [{
                            x: item.ratio,
                            y: item.momentum
                        }],
                        backgroundColor: item.color,
                        borderColor: item.color,
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        datalabels: {
                            align: 'right',
                            offset: 10,
                            formatter: function(value, context) {
                                return context.dataset.label;
                            }
                        }
                    };

                    if (showTail && item.tail_ratio && item.tail_ratio.length > 1) {
                        dataset.data = [];
                        for (let i = 0; i < item.tail_ratio.length; i++) {
                            dataset.data.push({
                                x: item.tail_ratio[i],
                                y: item.tail_momentum[i]
                            });
                        }

                        // First points are smaller
                        for (let i = 0; i < dataset.data.length - 1; i++) {
                            dataset.pointRadius = dataset.pointRadius || [];
                            dataset.pointRadius.push(4);
                        }

                        // Last point is larger
                        dataset.pointRadius.push(8);
                    }

                    return dataset;
                });

                // Create chart
                new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                type: 'linear',
                                position: 'center',
                                title: {
                                    display: true,
                                    text: 'RS-Ratio'
                                },
                                min: xRange[0],
                                max: xRange[1],
                                grid: {
                                    color: function(context) {
                                        return context.tick.value === 100 ? 'rgba(0, 0, 0, 0.5)' : 'rgba(0, 0, 0, 0.1)';
                                    },
                                    lineWidth: function(context) {
                                        return context.tick.value === 100 ? 2 : 1;
                                    }
                                }
                            },
                            y: {
                                type: 'linear',
                                position: 'center',
                                title: {
                                    display: true,
                                    text: 'RS-Momentum'
                                },
                                min: yRange[0],
                                max: yRange[1],
                                grid: {
                                    color: function(context) {
                                        return context.tick.value === 0 ? 'rgba(0, 0, 0, 0.5)' : 'rgba(0, 0, 0, 0.1)';
                                    },
                                    lineWidth: function(context) {
                                        return context.tick.value === 0 ? 2 : 1;
                                    }
                                }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const data = chartData.find(item => item.symbol === context.dataset.label);
                                        return [
                                            data.symbol,
                                            `RS-Ratio: ${context.parsed.x.toFixed(2)}`,
                                            `RS-Momentum: ${context.parsed.y.toFixed(2)}`,
                                            `Quadrant: ${data.quadrant}`
                                        ];
                                    }
                                }
                            },
                            legend: {
                                display: false
                            },
                            datalabels: {
                                color: '#000',
                                font: {
                                    weight: 'bold'
                                }
                            }
                        }
                    },
                    plugins: [ChartDataLabels]
                });
            });
        </script>
    <?php endif; ?>
</div>

<?php include('footer.php'); ?>