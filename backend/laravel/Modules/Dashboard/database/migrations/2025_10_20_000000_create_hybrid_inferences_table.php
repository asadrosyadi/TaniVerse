<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Unified output of the hybrid CNN-LSTM feature-fusion model (Algorithm 1).
 *
 * One row per edge inference timestep transmitted from the Raspberry Pi:
 * fused pest class + plant-health status + micro-climate/VPD + the actuation
 * decision. This replaces the modality-isolated pairing of `sensor_kameras`
 * (CNN only) and the Flask forecast (LSTM only).
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('hybrid_inferences', function (Blueprint $table) {
            $table->id();
            $table->string('iot_id');
            $table->string('image')->nullable();

            // CNN branch (soft-max over the fused representation)
            $table->string('pest_label')->nullable();       // e.g. "hispa" / "No pest detected"
            $table->string('pest_class')->nullable();
            $table->float('pest_confidence')->nullable();

            // LSTM / health branch
            $table->string('health_status')->nullable();     // Healthy | Moderate Stress | High Stress | Monitor
            $table->string('health_class')->nullable();
            $table->float('health_confidence')->nullable();

            // micro-climate regression head
            $table->float('vpd_kpa')->nullable();
            $table->json('microclimate')->nullable();         // {temperature, humidity, ph, light_intensity, vpd}

            // adaptive feature-fusion weights (Eq. 1)
            $table->float('fusion_w1')->nullable();
            $table->float('fusion_w2')->nullable();

            // Algorithm 1 step 15 actuation decision
            $table->string('actuation_action')->nullable();
            $table->json('actuators')->nullable();            // ["ultrasonic_repeller","uv_lamp"] etc.

            $table->string('transmitted_via')->nullable();    // "rest" | "rest+mqtt"
            $table->timestamps();

            $table->index(['iot_id', 'created_at']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('hybrid_inferences');
    }
};
