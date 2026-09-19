<?php

namespace Modules\Dashboard\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class HybridInference extends Model
{
    use HasFactory;

    protected $table = 'hybrid_inferences';

    protected $fillable = [
        'iot_id',
        'image',
        'pest_label',
        'pest_class',
        'pest_confidence',
        'health_status',
        'health_class',
        'health_confidence',
        'vpd_kpa',
        'microclimate',
        'fusion_w1',
        'fusion_w2',
        'actuation_action',
        'actuators',
        'transmitted_via',
    ];

    protected $casts = [
        'microclimate'      => 'array',
        'actuators'         => 'array',
        'pest_confidence'   => 'float',
        'health_confidence' => 'float',
        'vpd_kpa'           => 'float',
        'fusion_w1'         => 'float',
        'fusion_w2'         => 'float',
    ];
}
