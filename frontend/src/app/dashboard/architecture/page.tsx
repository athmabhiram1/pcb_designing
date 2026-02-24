"use client";

import React from "react";
import { motion } from "framer-motion";
import {
    BarChart3,
    Workflow,
    Cpu,
    Database,
    Globe,
    ShieldCheck,
    Layers,
    Zap,
    ChevronRight,
    Activity,
    Network,
    Terminal as TerminalIcon
} from "lucide-react";
import { MechButton } from "@/components/ui/MechButton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const pipelineSteps = [
    { id: "ING", name: "Ingestion", desc: "Natural language requirement parsing", icon: Globe, status: "Active" },
    { id: "SYN", name: "Synthesis", desc: "Generative architecture modeling", icon: Zap, status: "Active" },
    { id: "OPT", name: "Optimizer", desc: "Sub-millisecond routing kernels", icon: Cpu, status: "Active" },
    { id: "DFM", name: "Validator", desc: "DFM / ERC integrity check", icon: ShieldCheck, status: "Ready" },
    { id: "EXP", name: "Export", desc: "Production-ready netlists & BOMs", icon: Database, status: "Ready" },
];

const ArchitectureGrid = () => (
    <svg className="absolute inset-0 w-full h-full opacity-[0.03] pointer-events-none" xmlns="http://www.w3.org/2000/svg">
        <pattern id="archGrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" />
        </pattern>
        <rect width="100%" height="100%" fill="url(#archGrid)" />
    </svg>
);

export default function ArchitecturePage() {
    return (
        <div className="p-8 h-full flex flex-col gap-8 bg-dark-900 font-sans overflow-auto scrollbar-hide">
            {/* Header */}
            <header className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-primary font-mono text-[9px] uppercase tracking-[0.4em]">
                    <BarChart3 className="w-3.5 h-3.5" /> Neural Pipeline Architecture v0.9-alpha
                </div>
                <h1 className="text-4xl font-display font-black text-white uppercase tracking-tighter italic">Machine <span className="text-primary tracking-normal">Reasoning</span> Workflow</h1>
            </header>

            <div className="grid grid-cols-1 xl:grid-cols-4 gap-8 flex-1">
                {/* Visual Pipeline HUD */}
                <Card className="xl:col-span-3 bg-dark-800/40 border-white/5 relative overflow-hidden flex flex-col min-h-[600px]">
                    <ArchitectureGrid />
                    <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent pointer-events-none" />

                    <CardHeader className="p-8 border-b border-white/5 flex flex-row items-center justify-between z-10">
                        <div>
                            <CardTitle className="text-sm font-display font-bold text-white uppercase tracking-[0.2em]">Operational Pipeline</CardTitle>
                            <p className="text-[10px] text-white/20 italic mt-1 uppercase tracking-widest">End-to-End Generative Logic Flow</p>
                        </div>
                        <Badge variant="outline" className="bg-primary/5 border-primary/20 text-primary text-[10px] uppercase tracking-widest animate-pulse">Live Kernel</Badge>
                    </CardHeader>

                    <CardContent className="flex-1 p-12 flex flex-col justify-center items-center relative z-10">
                        <div className="flex flex-col gap-12 w-full max-w-2xl relative">
                            {/* Vertical Logic Line */}
                            <div className="absolute left-[31px] top-4 bottom-4 w-px bg-gradient-to-b from-primary/40 via-primary/10 to-transparent" />

                            {pipelineSteps.map((step, i) => (
                                <motion.div
                                    key={step.id}
                                    initial={{ x: -20, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="flex items-center gap-8 relative group"
                                >
                                    {/* Node Icon */}
                                    <div className="w-16 h-16 shrink-0 rounded-2xl bg-dark-900 border border-white/5 flex items-center justify-center relative shadow-2xl group-hover:border-primary/40 group-hover:scale-110 transition-all duration-500 z-10">
                                        <step.icon className="w-6 h-6 text-primary/60 group-hover:text-primary group-hover:drop-shadow-[0_0_8px_rgba(14,165,233,0.5)] transition-all" />
                                        <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-dark-900 border border-white/10 flex items-center justify-center text-[8px] font-mono text-white/40 font-bold italic">
                                            0{i + 1}
                                        </div>
                                    </div>

                                    {/* Step Details */}
                                    <div className="flex-1 p-5 rounded-2xl bg-white/[0.02] border border-white/5 group-hover:bg-white/[0.04] group-hover:border-white/10 transition-all translate-x-0 group-hover:translate-x-2">
                                        <div className="flex justify-between items-center mb-1">
                                            <h4 className="text-sm font-display font-black text-white uppercase tracking-widest">{step.name}</h4>
                                            <span className="text-[8px] font-mono text-white/20 uppercase tracking-widest px-1.5 py-0.5 rounded bg-dark-900">{step.status}</span>
                                        </div>
                                        <p className="text-[11px] text-white/40 italic font-light leading-relaxed">{step.desc}</p>
                                    </div>
                                </motion.div>
                            ))}

                            {/* Generative Visual Overlay */}
                            <div className="absolute -right-32 top-1/2 -translate-y-1/2 opacity-20 hidden 2xl:block">
                                <Activity className="w-64 h-64 text-primary" strokeWidth={0.5} />
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* technical Sidebar */}
                <div className="flex flex-col gap-6">
                    <Card className="bg-dark-800/60 border-white/5 p-6 space-y-6">
                        <div className="flex items-center gap-3">
                            <Workflow className="w-4 h-4 text-primary" />
                            <h3 className="text-[11px] font-display font-bold text-white uppercase tracking-[0.2em]">Sub-System Stack</h3>
                        </div>

                        <div className="space-y-4">
                            {[
                                { label: "Reasoning Engine", val: "Aether-Core v1.0", detail: "Transformers + Graph" },
                                { label: "Placement Kernel", val: "SA-PhysX 4.2", detail: "Simulated Annealing" },
                                { label: "Routing Latency", val: "142ms", detail: "Average per 1k nets" },
                                { label: "Memory Swap", val: "8.4GB", detail: "Temporal Vector DB" },
                            ].map((s, i) => (
                                <div key={i} className="flex flex-col gap-1 border-b border-white/5 pb-3 last:border-0 group cursor-default">
                                    <span className="text-[9px] text-white/20 uppercase tracking-widest group-hover:text-primary transition-colors">{s.label}</span>
                                    <span className="text-xs text-white/80 font-mono italic group-hover:text-white transition-colors">{s.val}</span>
                                    <span className="text-[8px] text-white/10 italic group-hover:text-white/30 transition-colors uppercase tracking-tight">{s.detail}</span>
                                </div>
                            ))}
                        </div>
                    </Card>

                    <Card className="bg-primary/5 border border-primary/10 p-6 flex flex-col gap-6 relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-2 opacity-10 rotate-12 group-hover:rotate-0 transition-transform duration-700">
                            <Network className="w-24 h-24 text-primary" />
                        </div>

                        <div className="relative z-10 flex flex-col gap-1">
                            <h3 className="text-[10px] font-mono text-primary font-bold uppercase tracking-[0.3em] mb-4">Neural Saturation</h3>
                            <div className="flex items-end gap-1.5 h-24 mb-4">
                                {[30, 60, 45, 90, 80, 55, 75, 40, 95, 65].map((h, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ height: 0 }}
                                        animate={{ height: `${h}%` }}
                                        transition={{ delay: i * 0.05, duration: 0.8 }}
                                        className="flex-1 bg-primary/20 rounded-t-sm group-hover:bg-primary/40 transition-colors"
                                    />
                                ))}
                            </div>
                            <div className="flex justify-between items-center text-[10px] font-mono text-primary/40 uppercase tracking-tighter">
                                <span>Cycle Velocity</span>
                                <span className="text-primary font-black">4.2 GHz</span>
                            </div>
                            <MechButton variant="primary" size="sm" className="mt-4 w-full text-[10px] font-black group">
                                Deep Diagnostics <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                            </MechButton>
                        </div>
                    </Card>

                    <Card className="bg-dark-800/40 border-white/5 p-4 flex gap-3 items-center hover:bg-dark-900 transition-colors cursor-pointer group">
                        <div className="w-8 h-8 rounded-lg bg-dark-900 flex items-center justify-center border border-white/5 group-hover:border-primary/20 transition-all">
                            <TerminalIcon className="w-4 h-4 text-white/20 group-hover:text-primary transition-colors" />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[9px] font-display font-black text-white/40 uppercase tracking-widest">Compiler Logs</span>
                            <span className="text-[8px] text-white/10 italic">0/12 Errors Detected</span>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
}
