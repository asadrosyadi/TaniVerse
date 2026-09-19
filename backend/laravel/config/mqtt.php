<?php

/**
 * MQTT broker settings for Algorithm 1 step 16 -- "Transmit the results to the
 * IoT Dashboard and Actuator System via Laravel API and MQTT Broker".
 *
 * App\Services\MqttPublisher publishes actuation payloads to
 *   {topic_prefix}/{iot_id}/inference
 * It uses php-mqtt/laravel-client when that package is installed, and otherwise
 * falls back to a minimal built-in QoS-0 publisher (no extra dependency).
 * Set MQTT_ENABLED=false to disable transmission entirely (REST only).
 */
return [
    'enabled'      => env('MQTT_ENABLED', false),
    'host'         => env('MQTT_HOST', '127.0.0.1'),
    'port'         => (int) env('MQTT_PORT', 1883),
    'username'     => env('MQTT_USERNAME'),
    'password'     => env('MQTT_PASSWORD'),
    'client_id'    => env('MQTT_CLIENT_ID', 'petaniasik-laravel'),
    'topic_prefix' => env('MQTT_TOPIC_PREFIX', 'taniverse'),
    'keepalive'    => (int) env('MQTT_KEEPALIVE', 60),
    'timeout'      => (int) env('MQTT_TIMEOUT', 3),
];
