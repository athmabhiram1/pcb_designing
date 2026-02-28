"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
    LibraryBig,
    Search,
    Cpu,
    Zap,
    ChevronRight,
    Plus,
    Loader2,
    Box,
    Network,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MechButton } from "@/components/ui/MechButton";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/lib/api";
import { usePlatform } from "@/lib/platform-context";

interface BackendTemplate {
    name: string;
    filename: string;
    description: string;
    component_count: number;
    net_count: number;
    category: string;
}

// ── Pretty name mapping ───────────────────────────────────────────────────
const prettyNames: Record<string, string> = {
    "555_timer": "555 Timer",
    "3v3_regulator": "3.3V Regulator",
    "led_resistor": "LED Resistor",
    "mosfet_switch": "MOSFET Switch",
    "opamp_buffer": "Op-Amp Buffer",
};

const icons: Record<string, React.ReactNode> = {
    "555_timer": <Zap className="w-5 h-5" />,
    "3v3_regulator": <Cpu className="w-5 h-5" />,
    "led_resistor": <Box className="w-5 h-5" />,
    "mosfet_switch": <Network className="w-5 h-5" />,
    "opamp_buffer": <Zap className="w-5 h-5" />,
};

const colors: Record<string, { border: string; text: string; bg: string }> = {
    "555_timer": { border: "border-orange-500/20", text: "text-orange-400", bg: "bg-orange-500/10" },
    "3v3_regulator": { border: "border-primary/20", text: "text-primary", bg: "bg-primary/10" },
    "led_resistor": { border: "border-green-500/20", text: "text-green-400", bg: "bg-green-500/10" },
    "mosfet_switch": { border: "border-pink-500/20", text: "text-pink-400", bg: "bg-pink-500/10" },
    "opamp_buffer": { border: "border-purple-500/20", text: "text-purple-400", bg: "bg-purple-500/10" },
};

const defaultColor = { border: "border-white/10", text: "text-white/40", bg: "bg-white/5" };

export default function TemplatesPage() {
    const [search, setSearch] = useState("");
    const [templates, setTemplates] = useState<BackendTemplate[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const router = useRouter();
    const { addLog } = usePlatform();

    useEffect(() => {
        api.templates()
            .then(data => { setTemplates(data); setLoading(false); })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, []);

    const filtered = templates.filter(t =>
        t.name.toLowerCase().includes(search.toLowerCase()) ||
        t.description.toLowerCase().includes(search.toLowerCase())
    );

    const handleInstantiate = (t: BackendTemplate) => {
        addLog({ time: new Date().toLocaleTimeString("en-US", { hour12: false }), type: "INFO", msg: `Loading template: ${prettyNames[t.name] || t.name}` });
        router.push(`/dashboard/generate`);
    };

    return (
        <div className="h-full flex flex-col bg-dark-900 font-sans overflow-hidden">
            {/* Header */}
            <header className="px-8 pt-7 pb-5 flex flex-col gap-5 z-20 shrink-0 border-b border-white/5 bg-dark-800/40 backdrop-blur-xl">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 text-primary font-mono text-[9px] uppercase tracking-[0.3em]">
                            <LibraryBig className="w-3 h-3" /> Schematic Archive
                        </div>
                        <div className="h-4 w-px bg-white/10" />
                        <h1 className="text-sm font-display font-black text-white uppercase tracking-tight">
                            Template <span className="text-primary">Library</span>
                        </h1>
                    </div>
                    <Badge variant="outline" className="text-[10px] border-white/10 text-white/30">{templates.length} available</Badge>
                </div>

                <div className="relative max-w-xl">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/15" />
                    <Input
                        placeholder="Search templates…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        className="pl-10 bg-dark-900 border-white/5 focus:border-primary/30 h-10 text-sm text-white placeholder:text-white/15"
                    />
                </div>
            </header>

            {/* Grid */}
            <ScrollArea className="flex-1 px-8 py-6">
                {loading ? (
                    <div className="flex items-center justify-center gap-3 text-white/25 py-32">
                        <Loader2 className="w-5 h-5 animate-spin text-primary" />
                        <span className="text-xs font-mono uppercase">Loading templates from backend…</span>
                    </div>
                ) : error ? (
                    <div className="text-center py-32">
                        <div className="text-white/30 text-sm mb-2">Backend unavailable</div>
                        <p className="text-white/15 text-xs font-mono">{error}</p>
                    </div>
                ) : filtered.length === 0 ? (
                    <div className="text-center py-32">
                        <p className="text-white/20 text-sm">No templates match "{search}".</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 pb-20">
                        {filtered.map((t, i) => {
                            const c = colors[t.name] || defaultColor;
                            return (
                                <motion.div key={t.name}
                                    initial={{ opacity: 0, y: 16 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.08 }}
                                >
                                    <Card className={cn(
                                        "bg-dark-800/40 border-white/5 hover:border-primary/30 transition-all group overflow-hidden relative cursor-pointer shadow-xl",
                                    )}
                                        onClick={() => handleInstantiate(t)}
                                    >
                                        <CardContent className="p-0">
                                            {/* Preview area */}
                                            <div className={cn("h-36 relative overflow-hidden", c.bg)}>
                                                <div className="absolute inset-0 bento-grid opacity-10 pointer-events-none" />
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <div className={cn("p-4 rounded-xl", c.bg, c.text)}>
                                                        {icons[t.name] || <Cpu className="w-5 h-5" />}
                                                    </div>
                                                </div>
                                                {t.category && (
                                                    <Badge className={cn("absolute top-3 right-3 text-[8px] uppercase tracking-widest font-bold border-none", c.bg, c.text)}>
                                                        {t.category}
                                                    </Badge>
                                                )}
                                            </div>

                                            {/* Info */}
                                            <div className="p-5 space-y-3">
                                                <h3 className="text-base font-display font-black text-white uppercase tracking-tight group-hover:text-primary transition-colors">
                                                    {prettyNames[t.name] || t.name.replace(/_/g, " ")}
                                                </h3>
                                                <p className="text-[11px] text-white/30 leading-relaxed line-clamp-2">{t.description}</p>

                                                <div className="flex gap-4 pt-2 font-mono text-[10px] text-white/20">
                                                    <span><span className="text-primary font-bold">{t.component_count}</span> components</span>
                                                    <span><span className="text-white/40 font-bold">{t.net_count}</span> nets</span>
                                                </div>

                                                <div className="flex justify-between items-center pt-3 border-t border-white/5">
                                                    <Badge variant="outline" className="text-[9px] border-white/10 text-white/25">{t.filename}</Badge>
                                                    <div className="flex items-center gap-1 text-primary text-[9px] font-bold uppercase group-hover:gap-2 transition-all">
                                                        Instantiate <ChevronRight className="w-3 h-3" />
                                                    </div>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                </motion.div>
                            );
                        })}

                        {/* Custom card */}
                        <motion.div
                            initial={{ opacity: 0, y: 16 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: filtered.length * 0.08 }}
                        >
                            <Card className="bg-dark-800/20 border-dashed border-white/10 hover:border-primary/30 transition-all group h-full flex cursor-pointer"
                                onClick={() => router.push("/dashboard/generate")}
                            >
                                <CardContent className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
                                    <div className="w-14 h-14 rounded-xl bg-primary/5 border border-primary/15 flex items-center justify-center group-hover:bg-primary/10 group-hover:border-primary/30 transition-all">
                                        <Plus className="w-6 h-6 text-primary/50 group-hover:text-primary transition-colors" />
                                    </div>
                                    <div className="text-center">
                                        <h3 className="text-xs font-display font-bold text-white/30 uppercase tracking-widest group-hover:text-white transition-colors">Design from Prompt</h3>
                                        <p className="text-[10px] text-white/15 mt-1">Use AI to generate a custom schematic</p>
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    </div>
                )}
            </ScrollArea>
        </div>
    );
}
