# scripts/runner.py

"""
Script principal para executar geração de mapeamentos.

Comandos disponíveis:
- single: Execução única com parâmetros específicos
- benchmark_v2: Benchmark completo configurável
"""

import os
import sys

# --- CORREÇÃO CRÍTICA ---
# Adiciona o diretório pai (raiz do projeto) ao caminho do Python IMEDIATAMENTE.
# Isso deve ser feito ANTES de importar qualquer coisa de 'mapping_generator'.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ------------------------

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Agora os imports vão funcionar
from mapping_generator.cli import create_parser
from mapping_generator.generation.controller import GenerationTask
from mapping_generator.utils.logger_setup import setup_logger

def run_single_generation(args):
    """
    Executa geração única baseada em argumentos CLI.
    """
    print("=" * 60)
    print("EXECUÇÃO SINGLE - Geração Única")
    print("=" * 60)
    
    task_params = _build_task_params_from_args(args)
    
    try:
        task = GenerationTask(**task_params)
        success = task.run()
        
        if success:
            print("\n✅ Geração concluída com sucesso!")
            # Mostrar onde salvou
            if task.generator and task.file_saver:
                print(f"📁 Resultados em: {os.path.abspath(task.file_saver.output_dir)}")
        else:
            print("\n❌ Geração falhou.")
            
    except Exception as e:
        print(f"\n❌ Erro durante geração: {e}")
        logging.error(f"Erro na task: {e}", exc_info=True)

def _build_task_params_from_args(args) -> dict:
    """Constrói dicionário de parâmetros."""
    params = {
        'tec': args.tec,
        'gen_mode': args.gen_mode,
        'k': args.k_graphs,
        'output_dir': args.output_dir,
        'no_images': args.no_images,
        'retries_multiplier': args.retries_multiplier,
        'visualize': args.visualize
    }
    
    if args.tec == 'cgra':
        params['arch_sizes'] = [tuple(args.arch_size)]
        params['cgra_params'] = {'bits': args.bits}
        params['graph_range'] = tuple(args.graph_range)
        params['k_range'] = tuple(args.k_range)
        params['no_extend_io'] = args.no_extend_io
        params['max_path_length'] = args.max_path_length
        params['ii'] = args.ii
        params['alpha'] = args.alpha
        
        if args.gen_mode == 'grammar':
            params['strategy'] = args.strategy
            if args.strategy == 'systematic':
                params['difficulty'] = args.difficulty
            elif args.strategy == 'random':
                if not args.difficulty_range:
                    raise ValueError("--difficulty-range obrigatório para --strategy random")
                params['difficulty_range'] = args.difficulty_range
                params['adaptive'] = True
    
    elif args.tec == 'qca':
        params['arch_sizes'] = [tuple(args.arch_size)]
        params['qca_arch'] = args.qca_arch
        params['num_inputs'] = args.num_inputs
        params['num_derivations'] = args.num_derivations
        params['routing_factor'] = args.routing_factor
        
        if hasattr(args, 'balanced') and args.balanced:
            params['balanced'] = True
        elif hasattr(args, 'unbalanced') and args.unbalanced:
            params['unbalanced'] = True
        else:
            params['balanced'] = True
    
    return params

def run_benchmark_v2(args):
    print("=" * 60)
    print("BENCHMARK V2")
    print("=" * 60)
    print("⚠️  Benchmark V2 is under development.")

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Configurar logger
    setup_logger(verbose=args.verbose)
    
    if args.command == 'single':
        run_single_generation(args)
    elif args.command == 'benchmark_v2':
        run_benchmark_v2(args)
    else:
        print(f"❌ Comando desconhecido: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
