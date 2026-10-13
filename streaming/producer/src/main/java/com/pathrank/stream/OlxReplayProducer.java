package com.pathrank.stream;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.kafka.clients.admin.Admin;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.NewTopic;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.Metric;
import org.apache.kafka.common.MetricName;
import org.apache.kafka.common.errors.TopicExistsException;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.example.GroupReadSupport;

/**
 * Replays OLX job events from Parquet into Kafka.
 *
 * <p>Input: events.parquet (user, item, event, timestamp), sorted by timestamp
 * (scripts/streaming.py to-parquet). Each row becomes one record on the topic:
 * key = user_id (so one user's events stay ordered on one partition),
 * value = {"user":..,"item":..,"event":"..","ts":..}. The Kafka record timestamp is the send time
 * (CreateTime), which the Spark job uses to measure Kafka -> S3 lag.
 *
 * <p>Idempotence is on (acks=all, bounded in-flight requests), so broker-side retries cannot
 * write a record twice. Writes a JSON summary (throughput, errors, producer metrics) to --out.
 *
 * <pre>
 *   java -jar target/olx-replay-producer.jar --parquet data/raw/olx/events.parquet \
 *        --topic job-events --partitions 12 --rate 0 --out results/streaming/producer.json
 * </pre>
 * --rate N   events/sec cap (0 = as fast as possible)    --limit N   stop after N events
 * --recreate-topic   delete + recreate the topic first   --create-only   only manage the topic
 * --dry-run          read and serialize, don't send (shows the Parquet reader is not the bottleneck)
 */
public final class OlxReplayProducer {

    private static final AtomicLong ACKED = new AtomicLong();
    private static final AtomicLong ERRORS = new AtomicLong();
    private static final AtomicReference<String> FIRST_ERROR = new AtomicReference<>();
    private static volatile boolean stop = false;
    private static volatile boolean finished = false;

    private OlxReplayProducer() {}

    public static void main(String[] argv) throws Exception {
        Map<String, String> a = parseArgs(argv);
        String parquet = a.getOrDefault("parquet", "data/raw/olx/events.parquet");
        String bootstrap = a.getOrDefault("bootstrap", "localhost:9092");
        String topic = a.getOrDefault("topic", "job-events");
        int partitions = Integer.parseInt(a.getOrDefault("partitions", "12"));
        long rate = Long.parseLong(a.getOrDefault("rate", "0"));
        long limit = Long.parseLong(a.getOrDefault("limit", "0"));
        String out = a.getOrDefault("out", "results/streaming/producer.json");
        String acks = a.getOrDefault("acks", "all");
        String compression = a.getOrDefault("compression", "lz4");
        int lingerMs = Integer.parseInt(a.getOrDefault("linger-ms", "10"));
        int batchSize = Integer.parseInt(a.getOrDefault("batch-size", String.valueOf(256 * 1024)));
        boolean dryRun = a.containsKey("dry-run");

        if (!dryRun) {
            ensureTopic(bootstrap, topic, partitions, a.containsKey("recreate-topic"));
            if (a.containsKey("create-only")) {
                return;
            }
        }

        Properties p = new Properties();
        p.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrap);
        p.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        p.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        p.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, "true");
        p.put(ProducerConfig.ACKS_CONFIG, acks);                       // idempotence requires all
        p.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, "5"); // <= 5 keeps ordering
        p.put(ProducerConfig.RETRIES_CONFIG, String.valueOf(Integer.MAX_VALUE));
        p.put(ProducerConfig.DELIVERY_TIMEOUT_MS_CONFIG, "120000");
        p.put(ProducerConfig.LINGER_MS_CONFIG, String.valueOf(lingerMs));
        p.put(ProducerConfig.BATCH_SIZE_CONFIG, String.valueOf(batchSize));
        p.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, compression);
        p.put(ProducerConfig.BUFFER_MEMORY_CONFIG, String.valueOf(256L * 1024 * 1024));
        p.put(ProducerConfig.CLIENT_ID_CONFIG, "olx-replay");

        Thread main = Thread.currentThread();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (finished) {
                return;
            }
            stop = true;                                    // Ctrl+C: stop reading, flush, write summary
            try {
                main.join(30_000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }));

        Callback cb = (md, ex) -> {
            if (ex == null) {
                ACKED.incrementAndGet();
            } else {
                ERRORS.incrementAndGet();
                FIRST_ERROR.compareAndSet(null, ex.toString());
            }
        };

        Configuration conf = new Configuration();
        long sent = 0;
        long bytes = 0;
        long minTs = Long.MAX_VALUE;
        long maxTs = Long.MIN_VALUE;
        Map<String, Object> metrics = new LinkedHashMap<>();
        long t0;
        long tSent;
        long t1;
        try (KafkaProducer<String, String> producer = dryRun ? null : new KafkaProducer<>(p);
             ParquetReader<Group> reader = ParquetReader.builder(new GroupReadSupport(), new Path(parquet))
                     .withConf(conf).build()) {
            StringBuilder sb = new StringBuilder(128);
            t0 = System.nanoTime();
            long lastLog = t0;
            Group g;
            while (!stop && (limit <= 0 || sent < limit) && (g = reader.read()) != null) {
                long user = g.getLong("user", 0);
                long item = g.getLong("item", 0);
                String event = g.getString("event", 0);
                long ts = g.getLong("timestamp", 0);
                sb.setLength(0);
                sb.append("{\"user\":").append(user).append(",\"item\":").append(item)
                  .append(",\"event\":\"").append(event).append("\",\"ts\":").append(ts).append('}');
                String key = Long.toString(user);
                String value = sb.toString();
                bytes += key.length() + value.length();               // ASCII only
                if (producer != null) {
                    producer.send(new ProducerRecord<>(topic, key, value), cb);
                } else {
                    ACKED.incrementAndGet();
                }
                sent++;
                minTs = Math.min(minTs, ts);
                maxTs = Math.max(maxTs, ts);

                if ((sent & 1023) == 0) {
                    long now = System.nanoTime();
                    if (rate > 0) {                                   // pace to the target rate
                        long due = t0 + (long) (sent * 1e9 / rate);
                        if (due > now) {
                            Thread.sleep((due - now) / 1_000_000, (int) ((due - now) % 1_000_000));
                        }
                    }
                    if (now - lastLog > 5_000_000_000L) {
                        double s = (now - t0) / 1e9;
                        System.err.printf("sent %,d  acked %,d  errors %d  %.0f events/s%n",
                                sent, ACKED.get(), ERRORS.get(), sent / s);
                        lastLog = now;
                    }
                }
            }
            tSent = System.nanoTime();
            if (producer != null) {
                producer.flush();                                     // wait for every ack
                for (Map.Entry<MetricName, ? extends Metric> e : producer.metrics().entrySet()) {
                    MetricName n = e.getKey();
                    if ("producer-metrics".equals(n.group()) && Set.of("record-send-rate",
                            "request-latency-avg", "request-latency-max", "batch-size-avg",
                            "compression-rate-avg", "records-per-request-avg", "record-retry-total")
                            .contains(n.name())) {
                        metrics.put(n.name(), e.getValue().metricValue());
                    }
                }
            }
            t1 = System.nanoTime();
        }

        double secs = (t1 - t0) / 1e9;
        long acked = ACKED.get();
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("parquet", parquet);
        r.put("topic", topic);
        r.put("partitions", partitions);
        r.put("dry_run", dryRun);
        r.put("acks", acks);
        r.put("idempotence", true);
        r.put("compression", compression);
        r.put("linger_ms", lingerMs);
        r.put("batch_size", batchSize);
        r.put("rate_limit", rate);
        r.put("sent", sent);
        r.put("acked", acked);
        r.put("errors", ERRORS.get());
        r.put("first_error", FIRST_ERROR.get());
        r.put("interrupted", stop);
        r.put("seconds", round(secs, 3));
        r.put("send_loop_seconds", round((tSent - t0) / 1e9, 3));
        r.put("events_per_sec", Math.round(acked / secs));
        r.put("mb_per_sec", round(bytes / secs / 1e6, 2));
        r.put("payload_bytes", bytes);
        r.put("min_event_ts", sent > 0 ? minTs : null);
        r.put("max_event_ts", sent > 0 ? maxTs : null);
        r.put("producer_metrics", metrics);
        r.put("java", System.getProperty("java.version"));
        r.put("cpus", Runtime.getRuntime().availableProcessors());
        String json = toJson(r);
        java.nio.file.Path outPath = Paths.get(out);
        if (outPath.getParent() != null) {
            Files.createDirectories(outPath.getParent());
        }
        Files.write(outPath, json.getBytes(StandardCharsets.UTF_8));
        System.out.println(json);
        finished = true;
        if (ERRORS.get() > 0 || acked != sent) {
            System.exit(1);
        }
    }

    /** Create the topic if missing; with recreate, delete it first and wait until it is gone. */
    static void ensureTopic(String bootstrap, String topic, int partitions, boolean recreate) throws Exception {
        try (Admin admin = Admin.create(Map.of(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrap))) {
            Set<String> names = admin.listTopics().names().get();
            if (names.contains(topic) && recreate) {
                admin.deleteTopics(List.of(topic)).all().get();
                for (int i = 0; i < 60 && admin.listTopics().names().get().contains(topic); i++) {
                    Thread.sleep(500);
                }
                names = admin.listTopics().names().get();
            }
            if (!names.contains(topic)) {
                for (int attempt = 0; ; attempt++) {
                    try {
                        admin.createTopics(List.of(new NewTopic(topic, partitions, (short) 1))).all().get();
                        System.err.printf("created topic %s with %d partitions%n", topic, partitions);
                        break;
                    } catch (ExecutionException e) {
                        // right after a delete the broker can still report "marked for deletion"
                        if (!(e.getCause() instanceof TopicExistsException) || attempt >= 30) {
                            throw e;
                        }
                        Thread.sleep(1000);
                    }
                }
            } else {
                int have = admin.describeTopics(List.of(topic)).allTopicNames().get().get(topic).partitions().size();
                System.err.printf("topic %s exists with %d partitions%n", topic, have);
            }
        }
    }

    static Map<String, String> parseArgs(String[] argv) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < argv.length; i++) {
            if (!argv[i].startsWith("--")) {
                throw new IllegalArgumentException("unexpected argument: " + argv[i]);
            }
            String k = argv[i].substring(2);
            if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
                m.put(k, argv[++i]);
            } else {
                m.put(k, "true");                                  // flag
            }
        }
        return m;
    }

    static double round(double x, int digits) {
        double f = Math.pow(10, digits);
        return Math.round(x * f) / f;
    }

    static String toJson(Object o) {
        if (o == null) {
            return "null";
        }
        if (o instanceof Map<?, ?> m) {
            StringBuilder sb = new StringBuilder("{\n");
            int i = 0;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                sb.append("  \"").append(e.getKey()).append("\": ").append(toJson(e.getValue()));
                sb.append(++i < m.size() ? ",\n" : "\n");
            }
            return sb.append('}').toString();
        }
        if (o instanceof Number || o instanceof Boolean) {
            if (o instanceof Double d && (d.isNaN() || d.isInfinite())) {
                return "null";
            }
            return o.toString();
        }
        return "\"" + o.toString().replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ") + "\"";
    }
}
