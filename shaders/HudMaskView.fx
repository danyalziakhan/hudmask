// Shows the HUDMASK texture from the HUD Mask add-on: the frame dimmed, and the
// mask in red on top of it. The HUD should be red and nothing else. Without the
// add-on, or in a frame with no HUD, nothing is red.

#include "ReShade.fxh"

texture TexHudMaskView : HUDMASK;
sampler sTexHudMaskView { Texture = TexHudMaskView; };

float3 PS_HudMaskView(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 c = tex2D(ReShade::BackBuffer, uv).rgb * 0.35;
    if (tex2Dsize(sTexHudMaskView).x < 2)
        return c;
    float a = saturate(tex2D(sTexHudMaskView, uv).a);
    return lerp(c, float3(1.0, 0.0, 0.0), a);
}

technique HudMaskView < ui_tooltip = "Shows the HUD Mask add-on's HUDMASK texture in red. For checking only; leave it off in play."; >
{
    pass { VertexShader = PostProcessVS; PixelShader = PS_HudMaskView; }
}
