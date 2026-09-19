<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;
use Throwable;

/**
 * Fire-and-forget MQTT publisher for actuation payloads (Algorithm 1 step 16).
 *
 * Order of preference:
 *   1. php-mqtt/laravel-client  (if installed)  -- robust, QoS configurable
 *   2. built-in raw-socket QoS-0 publisher      -- zero extra dependency
 *   3. no-op + Log::warning                     -- broker unreachable / disabled
 *
 * Publishing never throws: a failed transmit is logged and reported via the
 * boolean return so the REST response still succeeds.
 */
class MqttPublisher
{
    public function enabled(): bool
    {
        return (bool) config('mqtt.enabled', false);
    }

    public function topicFor(string $iotId, string $leaf = 'inference'): string
    {
        $prefix = trim((string) config('mqtt.topic_prefix', 'taniverse'), '/');
        return "{$prefix}/{$iotId}/{$leaf}";
    }

    /**
     * @return bool true if the payload was handed to the broker
     */
    public function publish(string $topic, array $payload): bool
    {
        if (! $this->enabled()) {
            return false;
        }

        $json = json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);

        try {
            if (class_exists(\PhpMqtt\Client\MqttClient::class)) {
                return $this->publishWithPackage($topic, $json);
            }
            return $this->publishWithSocket($topic, $json);
        } catch (Throwable $e) {
            Log::warning('MQTT publish failed', ['topic' => $topic, 'error' => $e->getMessage()]);
            return false;
        }
    }

    // ------------------------------------------------------------------ package
    private function publishWithPackage(string $topic, string $json): bool
    {
        $class = \PhpMqtt\Client\MqttClient::class;
        $client = new $class(
            (string) config('mqtt.host'),
            (int) config('mqtt.port'),
            (string) config('mqtt.client_id') . '-' . substr(md5(uniqid('', true)), 0, 6),
        );

        $settingsClass = \PhpMqtt\Client\ConnectionSettings::class;
        $settings = new $settingsClass();
        if (config('mqtt.username')) {
            $settings = $settings
                ->setUsername((string) config('mqtt.username'))
                ->setPassword((string) config('mqtt.password'));
        }
        $settings = $settings->setKeepAliveInterval((int) config('mqtt.keepalive', 60));

        $client->connect($settings, true);
        $client->publish($topic, $json, 0, false);
        $client->disconnect();

        return true;
    }

    // ------------------------------------------------------------------- socket
    /**
     * Minimal MQTT 3.1.1 CONNECT + PUBLISH (QoS 0) over a raw TCP socket.
     */
    private function publishWithSocket(string $topic, string $json): bool
    {
        $host    = (string) config('mqtt.host');
        $port    = (int) config('mqtt.port', 1883);
        $timeout = (int) config('mqtt.timeout', 3);

        $fp = @fsockopen($host, $port, $errno, $errstr, $timeout);
        if (! $fp) {
            Log::warning("MQTT socket connect failed: {$errstr} ({$errno})");
            return false;
        }
        stream_set_timeout($fp, $timeout);

        try {
            $clientId = (string) config('mqtt.client_id', 'petaniasik-laravel')
                . '-' . substr(md5(uniqid('', true)), 0, 6);

            // ---- CONNECT ----
            $protocol   = $this->encodeString('MQTT');
            $level      = chr(0x04);
            $user       = (string) config('mqtt.username');
            $pass       = (string) config('mqtt.password');
            $flags      = 0x02;                                 // clean session
            if ($user !== '') { $flags |= 0x80; }
            if ($pass !== '') { $flags |= 0x40; }
            $keepAlive  = pack('n', (int) config('mqtt.keepalive', 60));

            $payload = $this->encodeString($clientId);
            if ($user !== '') { $payload .= $this->encodeString($user); }
            if ($pass !== '') { $payload .= $this->encodeString($pass); }

            $variable = $protocol . $level . chr($flags) . $keepAlive;
            $body     = $variable . $payload;
            fwrite($fp, chr(0x10) . $this->encodeLength(strlen($body)) . $body);

            // read CONNACK (4 bytes); byte 3 == 0 means "accepted"
            $connack = fread($fp, 4);
            if (strlen($connack) < 4 || ord($connack[3]) !== 0) {
                Log::warning('MQTT CONNACK rejected or missing');
                return false;
            }

            // ---- PUBLISH (QoS 0) ----
            $pubBody = $this->encodeString($topic) . $json;
            fwrite($fp, chr(0x30) . $this->encodeLength(strlen($pubBody)) . $pubBody);

            // ---- DISCONNECT ----
            fwrite($fp, chr(0xE0) . chr(0x00));

            return true;
        } finally {
            fclose($fp);
        }
    }

    private function encodeString(string $s): string
    {
        return pack('n', strlen($s)) . $s;
    }

    /** MQTT "remaining length" variable-byte integer. */
    private function encodeLength(int $len): string
    {
        $out = '';
        do {
            $byte = $len % 128;
            $len  = intdiv($len, 128);
            if ($len > 0) {
                $byte |= 0x80;
            }
            $out .= chr($byte);
        } while ($len > 0);
        return $out;
    }
}
