<?php

use App\Http\Controllers\SensorController;
use Modules\Dashboard\Http\Controllers\HybridInferenceController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::post('/kirim_sensor', [SensorController::class, 'store']);
Route::get('/bacajson/{iot_id}', [SensorController::class, 'getDataLidar']);
Route::get('/sensor', [SensorController::class, 'getByiot_id']);

// Hybrid CNN-LSTM feature-fusion model (Algorithm 1, step 16)
Route::post('/inference', [HybridInferenceController::class, 'store']);
Route::get('/inference/{iot_id}', [HybridInferenceController::class, 'latest']);
