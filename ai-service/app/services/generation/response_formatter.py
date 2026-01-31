"""
Response Formatter - Standardized response formatting for all generators

Ensures consistent response structure across architecture, layout, and blockly generators.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.utils.logging import get_logger

logger = get_logger(__name__)


class ResponseFormatter:
    """
    Standardizes response format across all generators.
    
    Provides consistent metadata structure and error handling.
    """
    
    @staticmethod
    def format_success(
        data: Any,
        generator_type: str,
        metadata: Dict[str, Any],
        generation_time_ms: int
    ) -> Dict[str, Any]:
        """
        Format successful generation response.
        
        Args:
            data: Generated data (architecture/layout/blockly)
            generator_type: Type of generator ("architecture", "layout", "blockly")
            metadata: Generation metadata
            generation_time_ms: Total generation time in milliseconds
            
        Returns:
            Standardized response dictionary
        """
        
        response = {
            'success': True,
            'data': data,
            'metadata': {
                'generator_type': generator_type,
                'generation_time_ms': generation_time_ms,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                **metadata
            }
        }
        
        logger.debug(
            f"response.formatted.success",
            extra={
                "generator": generator_type,
                "generation_time_ms": generation_time_ms,
                "provider": metadata.get('provider', 'unknown')
            }
        )
        
        return response
    
    @staticmethod
    def format_error(
        error: Exception,
        generator_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            error: Exception that occurred
            generator_type: Type of generator
            metadata: Optional metadata about the attempt
            
        Returns:
            Standardized error response dictionary
        """
        
        response = {
            'success': False,
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'generator_type': generator_type
            },
            'metadata': metadata or {}
        }
        
        logger.error(
            f"response.formatted.error",
            extra={
                "generator": generator_type,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        )
        
        return response
    
    @staticmethod
    def format_partial(
        data: Any,
        generator_type: str,
        warnings: list,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format partial success response (with warnings).
        
        Args:
            data: Generated data
            generator_type: Type of generator
            warnings: List of warnings
            metadata: Generation metadata
            
        Returns:
            Standardized partial response dictionary
        """
        
        response = {
            'success': True,
            'partial': True,
            'data': data,
            'warnings': [
                {
                    'level': w.level,
                    'component': getattr(w, 'component', getattr(w, 'block_id', 'unknown')),
                    'message': w.message,
                    'suggestion': getattr(w, 'suggestion', '')
                }
                for w in warnings
            ],
            'metadata': {
                'generator_type': generator_type,
                'warning_count': len(warnings),
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                **metadata
            }
        }
        
        logger.warning(
            f"response.formatted.partial",
            extra={
                "generator": generator_type,
                "warning_count": len(warnings),
                "provider": metadata.get('provider', 'unknown')
            }
        )
        
        return response
    
    @staticmethod
    def extract_metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract summary information from metadata.
        
        Args:
            metadata: Full metadata dictionary
            
        Returns:
            Summarized metadata
        """
        
        return {
            'provider': metadata.get('provider', 'unknown'),
            'generation_method': metadata.get('generation_method', 'unknown'),
            'used_heuristic': metadata.get('used_heuristic', False),
            'tokens_used': metadata.get('tokens_used', 0),
            'api_duration_ms': metadata.get('api_duration_ms', 0),
            'validation_warnings': metadata.get('validation_warnings', 0)
        }
    
    @staticmethod
    def combine_responses(
        architecture_response: Dict[str, Any],
        layout_responses: Dict[str, Dict[str, Any]],
        blockly_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine all generator responses into complete response.
        
        Args:
            architecture_response: Architecture generation response
            layout_responses: Map of screen_id -> layout response
            blockly_response: Blockly generation response
            
        Returns:
            Combined complete response
        """
        
        # Check if all succeeded
        all_success = (
            architecture_response.get('success', False) and
            all(r.get('success', False) for r in layout_responses.values()) and
            blockly_response.get('success', False)
        )
        
        # Collect all warnings
        all_warnings = []
        
        if 'warnings' in architecture_response:
            all_warnings.extend(architecture_response['warnings'])
        
        for layout_resp in layout_responses.values():
            if 'warnings' in layout_resp:
                all_warnings.extend(layout_resp['warnings'])
        
        if 'warnings' in blockly_response:
            all_warnings.extend(blockly_response['warnings'])
        
        # Calculate total generation time
        total_time = (
            architecture_response.get('metadata', {}).get('generation_time_ms', 0) +
            sum(r.get('metadata', {}).get('generation_time_ms', 0) for r in layout_responses.values()) +
            blockly_response.get('metadata', {}).get('generation_time_ms', 0)
        )
        
        # Build combined response
        combined = {
            'success': all_success,
            'complete': all_success and len(all_warnings) == 0,
            'result': {
                'architecture': architecture_response.get('data'),
                'layouts': {
                    screen_id: resp.get('data')
                    for screen_id, resp in layout_responses.items()
                },
                'blockly': blockly_response.get('data')
            },
            'warnings': all_warnings,
            'metadata': {
                'total_generation_time_ms': total_time,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                'generators': {
                    'architecture': ResponseFormatter.extract_metadata_summary(
                        architecture_response.get('metadata', {})
                    ),
                    'layout': {
                        screen_id: ResponseFormatter.extract_metadata_summary(
                            resp.get('metadata', {})
                        )
                        for screen_id, resp in layout_responses.items()
                    },
                    'blockly': ResponseFormatter.extract_metadata_summary(
                        blockly_response.get('metadata', {})
                    )
                },
                'warning_count': len(all_warnings),
                'heuristic_used': any([
                    architecture_response.get('metadata', {}).get('used_heuristic', False),
                    any(r.get('metadata', {}).get('used_heuristic', False) for r in layout_responses.values()),
                    blockly_response.get('metadata', {}).get('used_heuristic', False)
                ])
            }
        }
        
        logger.info(
            "response.combined",
            extra={
                "success": all_success,
                "complete": combined['complete'],
                "total_time_ms": total_time,
                "warning_count": len(all_warnings),
                "heuristic_used": combined['metadata']['heuristic_used']
            }
        )
        
        return combined


# Global formatter instance
response_formatter = ResponseFormatter()