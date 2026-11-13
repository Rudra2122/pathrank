package com.pathrank.ads.flink;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.util.OutputTag;

/**
 * Kafka {@code ad-events} -> per-ad impression/click counts in event-time tumbling windows ->
 * Redis {@code ctr:<adId>}, which the ad server turns into pCTR for the auction.
 *
 * <pre>
 *   java $FLINK_OPENS -jar ctr-aggregator/target/ctr-aggregator.jar \
 *        --bootstrap localhost:9092 --topic ad-events --redis-host localhost \
 *        --window-ms 5000 --out-of-order-ms 2000 --wall-clock-fallback-ms 5000 --parallelism 4
 * </pre>
 *
 * Runs as an embedded local Flink cluster (no separate JobManager needed on a laptop).
 * Checkpoints every 10 s; Kafka offsets are committed on checkpoint, and the Redis sink is
 * idempotent per window, so a restart neither loses nor double counts a window.
 */
public final class CtrAggregatorJob {
    public static final OutputTag<AdEvent> LATE = new OutputTag<>("late-events") {};

    private CtrAggregatorJob() {}

    /** The windowing logic, separate from Kafka/Redis so the test can run it on a bounded list. */
    public static SingleOutputStreamOperator<WindowCount> windows(DataStream<String> raw, long windowMs,
                                                                  long outOfOrderMs, long wallClockFallbackMs) {
        return raw
                .flatMap(new ParseEvents()).name("parse-drop-malformed")
                .assignTimestampsAndWatermarks(WatermarkStrategy
                        .<AdEvent>forGenerator(ctx -> new WallClockFallbackWatermarks(outOfOrderMs, wallClockFallbackMs))
                        .withTimestampAssigner((e, t) -> e.ts))
                .keyBy(e -> e.adId)
                .window(TumblingEventTimeWindows.of(Duration.ofMillis(windowMs)))
                .sideOutputLateData(LATE)
                .aggregate(new CountAggregate(), new WindowResult()).name("count-per-ad-window");
    }

    public static void main(String[] argv) throws Exception {
        Map<String, String> a = parseArgs(argv);
        String bootstrap = a.getOrDefault("bootstrap", "localhost:9092");
        String topic = a.getOrDefault("topic", "ad-events");
        String group = a.getOrDefault("group", "ctr-aggregator");
        String redisHost = a.getOrDefault("redis-host", "localhost");
        int redisPort = Integer.parseInt(a.getOrDefault("redis-port", "6379"));
        long windowMs = Long.parseLong(a.getOrDefault("window-ms", "5000"));
        long outOfOrderMs = Long.parseLong(a.getOrDefault("out-of-order-ms", "2000"));
        long fallbackMs = Long.parseLong(a.getOrDefault("wall-clock-fallback-ms", "5000"));
        int parallelism = Integer.parseInt(a.getOrDefault("parallelism", "4"));
        boolean fromLatest = "latest".equals(a.getOrDefault("from", "earliest"));

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(parallelism);
        env.enableCheckpointing(10_000);
        env.getConfig().setAutoWatermarkInterval(200);

        KafkaSource<String> source = KafkaSource.<String>builder()
                .setBootstrapServers(bootstrap)
                .setTopics(topic)
                .setGroupId(group)
                .setStartingOffsets(fromLatest ? OffsetsInitializer.latest() : OffsetsInitializer.earliest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();

        DataStream<String> raw = env.fromSource(source, WatermarkStrategy.noWatermarks(), "kafka:" + topic);
        SingleOutputStreamOperator<WindowCount> counts = windows(raw, windowMs, outOfOrderMs, fallbackMs);

        counts.addSink(new RedisCtrSink(redisHost, redisPort)).name("redis-ctr");
        counts.getSideOutput(LATE).addSink(new CounterSink<>(redisHost, redisPort, "ads:late_events")).name("late-count");
        raw.filter(s -> AdEvent.parse(s) == null)
                .addSink(new CounterSink<>(redisHost, redisPort, "ads:malformed_events")).name("malformed-count");

        env.execute("pathrank-ctr-aggregator");
    }

    private static Map<String, String> parseArgs(String[] argv) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < argv.length; i++) {
            if (!argv[i].startsWith("--")) continue;
            String k = argv[i].substring(2);
            if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) m.put(k, argv[++i]);
            else m.put(k, "true");
        }
        return m;
    }
}
